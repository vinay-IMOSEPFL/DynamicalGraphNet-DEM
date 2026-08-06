import torch
from torch_geometric.data import Data


class SphereWallInteraction:
    """
    Constructs a PyTorch Geometric (PyG) graph representing spheres and their interactions 
    with each other and with flat boundaries (walls). 
    
    Boundary conditions are enforced using the "Ghost Node" method: real spheres are 
    reflected across the wall planes to create virtual "ghost" spheres. Collisions with 
    walls are mathematically treated as collisions with these ghost spheres.
    """
    def __init__(self, boundaries, threshold, device='cpu', dtype=torch.float32):
        """
        Initializes the interaction model by parsing the boundary plane equations.

        Args:
            boundaries (dict): A dictionary of plane equations, where each value is a 
                               tensor [A, B, C, D] representing Ax + By + Cz + D = 0.
            threshold (float): Cutoff distance for establishing an edge (interaction).
            device (str): Computation device ('cpu' or 'cuda').
            dtype (torch.dtype): Data type for the tensors.
        """
        self.device = device
        self.dtype = dtype
        self.threshold = threshold
        
        normals, offsets = [], []
        # Extract the normal vector [A, B, C] and the scalar offset [D] for each wall
        for plane in boundaries.values():
            A, B, C, D = plane.tolist()
            normals.append([A, B, C])
            offsets.append(D)
            
        self.normals = torch.tensor(normals, dtype=dtype, device=device)
        self.offsets = torch.tensor(offsets, dtype=dtype, device=device)

    def insert_sphere_sphere_edges(self, sphere_positions, sphere_sphere_dist_matrix):
        """
        Builds graph edges based on a pairwise distance matrix.

        Args:
            sphere_positions (torch.Tensor): Tensor of shape (N, 3) holding 3D coordinates.
            sphere_sphere_dist_matrix (torch.Tensor): Tensor of shape (N, N) holding pairwise distances.

        Returns:
            tuple: (bidirectional_edges, bidirectional_edge_attrs)
                   where edges is shape (2, E) and attrs is shape (E, 3) holding distance vectors.
        """
        device = sphere_sphere_dist_matrix.device
        
        # Find pairs within the interaction threshold.
        # torch.triu(..., diagonal=1) ensures we only look at the upper triangle of the matrix 
        # to avoid duplicating edges (i->j and j->i) and ignores self-loops (diagonal=0).
        edge_indices = ((sphere_sphere_dist_matrix < self.threshold) & 
                        (torch.triu(torch.ones_like(sphere_sphere_dist_matrix), diagonal=1) == 1))
        
        # Convert boolean mask to list of index pairs [2, num_edges]
        edge_indices = edge_indices.nonzero(as_tuple=False).t()

        # Handle edge case: No particles are close enough to interact
        if edge_indices.numel() == 0:
            return (torch.empty((0, 2), dtype=torch.long, device=device),
                    torch.empty((0, 3), dtype=self.dtype, device=device))

        # The edge attribute is the spatial distance vector pointing from node i to node j
        edge_attributes = sphere_positions[edge_indices[1]] - sphere_positions[edge_indices[0]]

        # Duplicate the edges to make the graph undirected (bidirectional)
        # Concat (i->j) with (j->i)
        bidirectional_edges = torch.cat([edge_indices, edge_indices[[1, 0], :]], dim=1)
        # Concat vectors with their inverse directions
        bidirectional_edge_attrs = torch.cat([edge_attributes, -edge_attributes], dim=0)
        
        return (bidirectional_edges, bidirectional_edge_attrs)

    def reflect(self, pos):
        """
        Geometrically reflects all real spheres across all 6 cuboid walls to create ghost spheres.

        Args:
            pos (torch.Tensor): Real sphere positions of shape (N, 3).

        Returns:
            tuple: (all_pos, all_node_type) 
                   all_pos includes real + ghost positions.
                   all_node_type is 0 for real, 1 for ghost.
        """
        N = pos.size(0)
        device = pos.device
        
        # Expand positions to calculate reflections against all 6 walls simultaneously
        # p shape becomes (6, N, 3)
        p = pos.unsqueeze(0).expand(6, -1, -1).to(device)
        norm_n = self.normals.norm(dim=1, keepdim=True).to(device)
        
        # Calculate signed distance 'd' from each point to each plane: d = (P dot N) + D
        d = (p * self.normals.unsqueeze(1)).sum(dim=2) + self.offsets.unsqueeze(1).to(device)
        n_unit = self.normals / norm_n
        
        # Reflection formula: P_refl = P - 2 * (d / |N|) * N_unit
        coeff = (d / norm_n).unsqueeze(-1)
        p_refl = (p - 2 * coeff * n_unit.unsqueeze(1)).reshape(-1, 3)

        # Concatenate real positions with the newly generated ghost positions
        all_pos = torch.cat([pos, p_refl], dim=0) 
        
        # Create labels: 0 denotes a real sphere, 1 denotes a ghost (wall) sphere
        node_type = torch.zeros(N, 1).to(device)
        node_type_r = torch.ones_like(p_refl[:, 0:1])
        all_node_type = torch.cat([node_type, node_type_r], dim=0) 
        
        return all_pos, all_node_type          

    def process(self, sphere_positions):
        """
        Executes the full boundary processing pipeline: 
        1. Generates ghosts. 
        2. Computes all pairwise distances. 
        3. Creates edges. 
        4. Filters out irrelevant ghost-ghost edges.

        Args:
            sphere_positions (torch.Tensor): Original real positions.

        Returns:
            tuple: Filtered edge_index, edge_attr, all positions, and node type labels.
        """
        all_pos, all_node_type = self.reflect(sphere_positions.to(self.device, dtype=self.dtype))
        
        # Compute dense distance matrix for all nodes (Real + Ghosts)
        D = torch.cdist(all_pos, all_pos)
        edge_index, edge_attr = self.insert_sphere_sphere_edges(all_pos, D)
    
        # Safe fallback if completely sparse
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_attr = torch.empty((0, edge_attr.shape[1] if edge_attr.ndim > 1 else 1), dtype=edge_attr.dtype, device=self.device)
            return edge_index, edge_attr, all_pos, all_node_type
    
        # Ghosts are only used to exert forces on real spheres.
        # We must delete any edge that connects a ghost to another ghost.
        nt = all_node_type.squeeze(1)
        src, dst = edge_index
        keep = ~((nt[src] == 1) & (nt[dst] == 1)) # Keep if NOT (src is ghost AND dst is ghost)
    
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]
    
        return edge_index, edge_attr, all_pos, all_node_type

    def create_graph(self, pos, vel_t, vel_tm1, ang_vel_t, ang_vel_tm1, pos_tp1, vel_tp1, ang_vel_tp1):
        """
        Top-level builder method. Takes raw kinematic data from the dataset, runs the 
        boundary processing to construct the graph topology, pads the ghost kinematics, 
        and packages it into a PyTorch Geometric Data object.
        """
        # Ensure all inputs are floated and pushed to the target device
        pos    = pos.float().to(self.device, dtype=self.dtype)
        vel_t  = vel_t.float().to(self.device, dtype=self.dtype)
        vel_tm1  = vel_tm1.float().to(self.device, dtype=self.dtype)
        ang_vel_t = ang_vel_t.float().to(self.device, dtype=self.dtype)
        ang_vel_tm1 = ang_vel_tm1.float().to(self.device, dtype=self.dtype)
        
        pos_tp1 = pos_tp1.float().to(self.device, dtype=self.dtype)
        vel_tp1 = vel_tp1.float().to(self.device, dtype=self.dtype)
        ang_vel_tp1 = ang_vel_tp1.float().to(self.device, dtype=self.dtype)

        # Get topology and ghost node features
        edge_index, edge_attr, all_pos, all_node_type = self.process(pos)
        
        # Initialize full state tensors for real + ghost nodes (all zeros by default)
        all_vel = torch.zeros_like(all_pos)
        all_prev_vel = torch.zeros_like(all_pos)
        all_angvel = torch.zeros_like(all_pos)
        all_prev_angvel = torch.zeros_like(all_pos)
        
        # Mask to identify real nodes
        mask_b = (all_node_type == 0).squeeze()

        # Inject real node kinematics. 
        # Note: Ghost velocities remain exactly 0 since these boundaries are stationary flat walls.
        all_vel[mask_b] = vel_t
        all_prev_vel[mask_b] = vel_tm1
        all_angvel[mask_b] = ang_vel_t
        all_prev_angvel[mask_b] = ang_vel_tm1
        
        # Package into PyG Data object
        graph = Data(edge_index=edge_index, edge_attr=edge_attr)
        graph.node_feat = all_node_type
        graph.pos = all_pos
        graph.vel = all_vel
        graph.prev_vel = all_prev_vel
        graph.ang_vel = all_angvel
        graph.prev_ang_vel = all_prev_angvel
        
        # Targets for supervised training (Ground Truth displacements/changes)
        graph.y_dx = pos_tp1 - pos
        graph.y_dv = vel_tp1 - vel_t
        graph.y_dw = ang_vel_tp1 - ang_vel_t
        
        return graph
    
def insert_boundary(geo_data, device='cpu', dtype=torch.float32):
    """
    Generates the plane equations (Ax + By + Cz + D = 0) for a standard 3D cuboid.

    Args:
        geo_data (tuple): Contains nested tuples ((xmin, ymin, zmin), (xmax, ymax, zmax)).
        device (str): Computation device.
        dtype (torch.dtype): Tensor data type.

    Returns:
        dict: Keys represent face names, values are parameter tensors [A, B, C, D].
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = geo_data
    
    # Plane format: [nx, ny, nz, d] where nx*X + ny*Y + nz*Z + d = 0
    return {
        'front':  torch.tensor([ 1,  0,  0, -xmax], dtype=dtype, device=device), # x = xmax -> x - xmax = 0
        'back':   torch.tensor([-1,  0,  0,  xmin], dtype=dtype, device=device), # x = xmin -> -x + xmin = 0
        'right':  torch.tensor([ 0,  1,  0, -ymax], dtype=dtype, device=device),
        'left':   torch.tensor([ 0, -1,  0,  ymin], dtype=dtype, device=device),
        'top':    torch.tensor([ 0,  0,  1, -zmax], dtype=dtype, device=device),
        'bottom': torch.tensor([ 0,  0, -1,  zmin], dtype=dtype, device=device)
    }
# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE (EPFL),
# Switzerland, 2025.
# Authors: Vinay Sharma and Olga Fink, Laboratory of Intelligent Maintenance and
# Operations Systems (IMOS).
# Released under the Non-Commercial License Agreement in LICENSE.txt.

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

    def _bidirectional(self, edges, attrs):
        """Mirror an edge list so every pair appears in both directions."""
        return (torch.cat([edges, edges.flip(0)], dim=1),
                torch.cat([attrs, -attrs], dim=0))

    def insert_sphere_sphere_edges(self, sphere_positions):
        """
        Edges between real spheres closer than the interaction threshold.

        Returns edge_index (2, E) and edge_attr (E, 3), the vector from source to target.
        """
        D = torch.cdist(sphere_positions, sphere_positions)

        # Upper triangle only: excludes self-loops and avoids emitting each pair twice.
        pairs = (D < self.threshold) & (torch.triu(torch.ones_like(D), diagonal=1) == 1)
        edges = pairs.nonzero(as_tuple=False).t()

        if edges.numel() == 0:
            return (torch.empty((2, 0), dtype=torch.long, device=self.device),
                    torch.empty((0, 3), dtype=self.dtype, device=self.device))

        attrs = sphere_positions[edges[1]] - sphere_positions[edges[0]]
        return self._bidirectional(edges, attrs)

    def insert_sphere_wall_edges(self, sphere_positions, all_pos):
        """
        Edges between each sphere and its own reflections, when within the threshold.

        A ghost stands for its parent's contact with one wall, so it connects to that sphere
        and nothing else. Ghosts are stacked one block of N per wall, so the ghost at block
        offset k mirrors sphere k % N.
        """
        N = sphere_positions.size(0)
        n_walls = (all_pos.size(0) - N) // N

        parent = torch.arange(N, device=self.device).repeat(n_walls)
        ghost = torch.arange(N, N * (n_walls + 1), device=self.device)

        # Vector from a sphere to its image; its length is twice the distance to the wall.
        attrs = all_pos[ghost] - sphere_positions[parent]
        within = attrs.norm(dim=1) < self.threshold

        if not within.any():
            return (torch.empty((2, 0), dtype=torch.long, device=self.device),
                    torch.empty((0, 3), dtype=self.dtype, device=self.device))

        edges = torch.stack([parent[within], ghost[within]], dim=0)
        return self._bidirectional(edges, attrs[within])

    def reflect(self, pos):
        """
        Places one ghost sphere per wall, beyond that wall, for every real sphere.

        The ghost sits |d| past the plane, d being the signed distance from the sphere centre.
        Inside the enclosure this is the mirror image, and the separation 2|d| reproduces a
        symmetric two-body collision against an immovable wall.

        Using |d| rather than the signed d keeps the ghost outside once a sphere crosses the
        plane; with the signed form the ghost would swap to the interior and the contact would
        push the sphere out of the domain instead of back into it. The two agree for d < 0.

        Args:
            pos (torch.Tensor): Real sphere positions of shape (N, 3).

        Returns:
            tuple: (all_pos, all_node_type)
                   all_pos stacks the N real positions with 6*N ghosts, one block of N per wall.
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

        # Ghost at P + 2*|d|*n_unit: always beyond the wall, so contact always repels inward.
        # Normals point out of the enclosure, so this reduces to the mirror image when d < 0.
        coeff = (d / norm_n).unsqueeze(-1)
        p_refl = (p + 2 * coeff.abs() * n_unit.unsqueeze(1)).reshape(-1, 3)

        # Concatenate real positions with the newly generated ghost positions
        all_pos = torch.cat([pos, p_refl], dim=0)

        # Create labels: 0 denotes a real sphere, 1 denotes a ghost (wall) sphere
        node_type = torch.zeros(N, 1).to(device)
        node_type_r = torch.ones_like(p_refl[:, 0:1])
        all_node_type = torch.cat([node_type, node_type_r], dim=0)

        return all_pos, all_node_type


    def process(self, sphere_positions):
        """
        Builds the topology in two passes: sphere-sphere edges among the real particles,
        then sphere-wall edges linking each sphere to its own reflections. Ghost-ghost and
        ghost-to-other-sphere edges cannot arise, so no filtering is needed.

        Returns edge_index, edge_attr, all positions, and node type labels.
        """
        pos = sphere_positions.to(self.device, dtype=self.dtype)

        ss_edges, ss_attrs = self.insert_sphere_sphere_edges(pos)

        all_pos, all_node_type = self.reflect(pos)
        sw_edges, sw_attrs = self.insert_sphere_wall_edges(pos, all_pos)

        edge_index = torch.cat([ss_edges, sw_edges], dim=1)
        edge_attr = torch.cat([ss_attrs, sw_attrs], dim=0)

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

        # Ghosts are stationary: the walls do not move, so the contact relative velocity is
        # the sphere's own. Giving a ghost its parent's mirrored velocity instead would make
        # v_sphere - v_ghost = 2(v.n)n, purely normal, cancelling the tangential slip that
        # generates spin on an oblique impact.
        n_ghost = all_pos.size(0) - pos.size(0)
        zeros = torch.zeros(n_ghost, 3, dtype=self.dtype, device=self.device)
        all_vel = torch.cat([vel_t, zeros], dim=0)
        all_prev_vel = torch.cat([vel_tm1, zeros], dim=0)
        all_angvel = torch.cat([ang_vel_t, zeros], dim=0)
        all_prev_angvel = torch.cat([ang_vel_tm1, zeros], dim=0)

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
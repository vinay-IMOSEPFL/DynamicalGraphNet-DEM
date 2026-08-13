import torch
import numpy as np
from torch_geometric.data import Data

class SphereWallInteraction:
    """
    Constructs a PyTorch Geometric (PyG) graph representing spheres and their interactions
    with each other and with flat boundaries (walls) in a cuboidal enclosure.

    Boundary conditions are enforced using the "Ghost Node" method: real spheres are
    reflected across the wall planes to create virtual "ghost" spheres.
    """
    def __init__(self, boundaries, threshold, device='cpu', dtype=torch.float32):
        """
        Initializes the interaction model by parsing the 6 boundary plane equations.

        Args:
            boundaries (dict): Plane equations [A, B, C, D] for each wall.
            threshold (float): Cutoff distance for establishing an edge.
            device (str): Computation device.
            dtype (torch.dtype): Tensor data type.
        """
        self.device = device
        self.dtype = dtype
        self.threshold = float(threshold)

        normals, offsets = [], []
        # Extract the normal vector [A, B, C] and the scalar offset [D] for each plane
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
        """Edges between real spheres closer than the interaction threshold."""
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
        Edges between each sphere and its own reflections, where the two are within
        the interaction threshold. Ghosts are stacked one block of N per wall, so the
        ghost at block offset k mirrors sphere k % N.
        """
        N = sphere_positions.size(0)
        n_walls = (all_pos.size(0) - N) // N

        parent = torch.arange(N, device=self.device).repeat(n_walls)
        ghost = torch.arange(N, N * (n_walls + 1), device=self.device)

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
        """
        N = pos.size(0)

        # Expand positions to calculate against all 6 walls simultaneously (6, N, 3)
        p = pos.unsqueeze(0).expand(6, -1, -1).to(self.device)
        norm_n = self.normals.norm(dim=1, keepdim=True).to(self.device)

        # Calculate signed distance 'd' from each point to each plane
        d = (p * self.normals.unsqueeze(1)).sum(dim=2) + self.offsets.unsqueeze(1).to(self.device)
        n_unit = self.normals / norm_n
        coeff = (d / norm_n).unsqueeze(-1)

        # Ghost at P + 2*|d|*n_unit: always beyond the wall, so contact always repels inward.
        p_refl = (p + 2 * coeff.abs() * n_unit.unsqueeze(1)).reshape(-1, 3)

        # Concatenate original positions with all ghost positions
        all_pos = torch.cat([pos, p_refl], dim=0)

        # 0 = real sphere, 1 = ghost sphere
        node_type = torch.zeros(N, 1, device=self.device, dtype=self.dtype)
        node_type_r = torch.ones_like(p_refl[:, 0:1])
        all_node_type = torch.cat([node_type, node_type_r], dim=0)

        return all_pos, all_node_type


    def process(self, sphere_positions):
        """
        Builds sphere-sphere edges among the real particles, then sphere-wall edges
        linking each sphere to its own reflections. Ghost-ghost and ghost-to-other-sphere
        edges cannot arise, so no filtering is needed.
        """
        pos = sphere_positions.to(self.device, dtype=self.dtype)

        ss_edges, ss_attrs = self.insert_sphere_sphere_edges(pos)

        all_pos, all_node_type = self.reflect(pos)
        sw_edges, sw_attrs = self.insert_sphere_wall_edges(pos, all_pos)

        return (torch.cat([ss_edges, sw_edges], dim=1),
                torch.cat([ss_attrs, sw_attrs], dim=0),
                all_pos, all_node_type)

    def create_graph(self, pos, vel_t, vel_tm1, ang_vel_t, ang_vel_tm1, pos_tp1, vel_tp1, ang_vel_tp1):
        """Constructs the PyTorch Geometric Data graph object."""
        to = lambda t: t.float().to(self.device, dtype=self.dtype)
        pos, vel_t, vel_tm1 = to(pos), to(vel_t), to(vel_tm1)
        ang_vel_t, ang_vel_tm1 = to(ang_vel_t), to(ang_vel_tm1)
        pos_tp1, vel_tp1, ang_vel_tp1 = to(pos_tp1), to(vel_tp1), to(ang_vel_tp1)

        edge_index, edge_attr, all_pos, all_node_type = self.process(pos)

        # Ghosts are stationary: the walls do not move, so the contact relative velocity is
        # the sphere's own. A mirrored ghost velocity would cancel the tangential slip that
        # generates spin on an oblique impact.
        n_ghost = all_pos.size(0) - pos.size(0)
        zeros = torch.zeros(n_ghost, 3, dtype=self.dtype, device=self.device)
        all_vel = torch.cat([vel_t, zeros], dim=0)
        all_prev_vel = torch.cat([vel_tm1, zeros], dim=0)
        all_angvel = torch.cat([ang_vel_t, zeros], dim=0)
        all_prev_angvel = torch.cat([ang_vel_tm1, zeros], dim=0)

        graph = Data(edge_index=edge_index, edge_attr=edge_attr)
        graph.node_feat = all_node_type
        graph.pos = all_pos
        graph.vel = all_vel
        graph.prev_vel = all_prev_vel
        graph.ang_vel = all_angvel
        graph.prev_ang_vel = all_prev_angvel
        graph.y_dx = pos_tp1 - pos
        graph.y_dv = vel_tp1 - vel_t
        graph.y_dw = ang_vel_tp1 - ang_vel_t

        return graph


class CylinderInteraction:
    """
    Ghost-node graph builder for spheres inside a rotating cylindrical drum.

    Reflections must be computed across three distinct surfaces:
    1. The curved cylindrical wall (radially).
    2. The top flat circular cap (axially).
    3. The bottom flat circular cap (axially).
    """
    def __init__(self, boundaries, threshold, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.threshold = float(threshold)

        # Cylinder geometry definition
        self.center = torch.as_tensor(boundaries["center"], device=device, dtype=dtype)
        axis = torch.as_tensor(boundaries["axis"], device=device, dtype=dtype)
        self.axis = axis / (axis.norm() + 1e-12) # Ensure the main axis is a unit vector
        self.radius = torch.as_tensor(boundaries["radius"], device=device, dtype=dtype)
        self.length = torch.as_tensor(boundaries["length"], device=device, dtype=dtype)
        self.halfL = 0.5 * self.length

    def _reflect_surface(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Mirrors points across the infinite curved cylindrical surface.
        """
        a, c, R, eps = self.axis, self.center, self.radius, 1e-9

        # 1. Project the particle's position onto the central axis
        v = pos - c
        t = v @ a
        p_ax = c + t.unsqueeze(1) * a # Closest point on the central axis

        # 2. Calculate the outward radial vector from the axis to the particle
        r = pos - p_ax
        rn = r.norm(dim=1, keepdim=True)
        r_hat = r / (rn + eps) # Unit radial vector

        # 3. Edge Case Handling: If a particle is exactly on the center axis, its radial
        #    vector is zero. We must supply an arbitrary perpendicular vector to reflect it.
        on_ax = rn.squeeze(1) < 1e-6
        if on_ax.any():
            ref = torch.tensor([1.0, 0.0, 0.0], device=pos.device, dtype=pos.dtype)
            # Prevent collinearity issues if the main axis happens to be the X-axis
            if torch.abs((a * ref).sum()) > 0.99:
                ref = torch.tensor([0.0, 1.0, 0.0], device=pos.device, dtype=pos.dtype)
            perp = torch.cross(a.expand_as(pos[on_ax]), ref.expand_as(pos[on_ax]))
            r_hat[on_ax] = perp / (perp.norm(dim=1, keepdim=True) + eps)

        # 4. Ghost at twice the gap to the wall, always on the far side. The absolute value
        #    keeps it outside once a particle passes the surface; with the signed gap the
        #    ghost would move inward and the contact would push the particle out of the drum.
        return pos + 2.0 * (R - rn).abs() * r_hat

    def _reflect_cap(self, pos: torch.Tensor, sign: float = 1.0) -> torch.Tensor:
        """
        Mirrors points across the flat top (sign=1) or bottom (sign=-1) caps of the cylinder.
        """
        # Center point of the cap
        cap_c = self.center + (sign * self.halfL) * self.axis
        n = sign * self.axis # Outward facing normal vector of the cap

        # Ghost at |d| beyond the cap, on the far side for either sign of d, as for the
        # cuboid walls. The two forms agree while the particle is inside (d < 0).
        d = ((pos - cap_c) @ n).unsqueeze(1)
        return pos + 2.0 * d.abs() * n

    def reflect(self, pos: torch.Tensor):
        """
        Aggregates all three reflection surfaces to generate the complete boundary set.
        """
        pos = pos.to(self.device, dtype=self.dtype)
        N = pos.size(0)

        # Combine the curved wall ghosts and both end-cap ghosts
        refl = torch.cat([self._reflect_surface(pos),
                          self._reflect_cap(pos, +1.0),
                          self._reflect_cap(pos, -1.0)], dim=0)

        all_pos = torch.cat([pos, refl], dim=0)

        # Create labels: 0 for the N real spheres, 1 for the 3*N ghost spheres
        all_nt = torch.cat([torch.zeros(N, 1, device=pos.device, dtype=self.dtype),
                            torch.ones(3 * N, 1, device=pos.device, dtype=self.dtype)], dim=0)

        return all_pos, all_nt

    def _bidirectional(self, edges, attrs):
        """Mirror an edge list so every pair appears in both directions."""
        return (torch.cat([edges, edges.flip(0)], dim=1),
                torch.cat([attrs, -attrs], dim=0))

    def process(self, sphere_positions: torch.Tensor):
        """
        Builds sphere-sphere edges among the real particles, then sphere-wall edges
        linking each sphere to its own reflections in the curved wall and the two caps.
        Ghost-ghost and ghost-to-other-sphere edges cannot arise.
        """
        pos = sphere_positions.to(self.device, dtype=self.dtype)
        N = pos.size(0)

        # 1. Sphere-sphere
        D = torch.cdist(pos, pos)
        pairs = (D < self.threshold) & torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        ss = pairs.nonzero(as_tuple=False).t()
        if ss.numel() == 0:
            ss_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
            ss_attrs = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        else:
            ss_edges, ss_attrs = self._bidirectional(ss, pos[ss[1]] - pos[ss[0]])

        # 2. Sphere-wall: each ghost bonds only with the sphere it mirrors.
        all_pos, all_nt = self.reflect(pos)
        n_surfaces = (all_pos.size(0) - N) // N
        parent = torch.arange(N, device=self.device).repeat(n_surfaces)
        ghost = torch.arange(N, N * (n_surfaces + 1), device=self.device)

        attrs = all_pos[ghost] - pos[parent]
        within = attrs.norm(dim=1) < self.threshold
        if not within.any():
            sw_edges = torch.empty((2, 0), dtype=torch.long, device=self.device)
            sw_attrs = torch.empty((0, 3), dtype=self.dtype, device=self.device)
        else:
            sw_edges, sw_attrs = self._bidirectional(
                torch.stack([parent[within], ghost[within]], dim=0), attrs[within])

        return (torch.cat([ss_edges, sw_edges], dim=1),
                torch.cat([ss_attrs, sw_attrs], dim=0),
                all_pos, all_nt)

    def create_graph(self, pos, vel_t, vel_tm1, ang_vel_t, ang_vel_tm1, pos_tp1, vel_tp1, ang_vel_tp1) -> Data:
        """Constructs the PyTorch Geometric Data object for the cylinder."""
        to = lambda t: t.float().to(self.device, dtype=self.dtype)
        pos, vel_t, vel_tm1 = to(pos), to(vel_t), to(vel_tm1)
        ang_vel_t, ang_vel_tm1 = to(ang_vel_t), to(ang_vel_tm1)
        pos_tp1, vel_tp1, ang_vel_tp1 = to(pos_tp1), to(vel_tp1), to(ang_vel_tp1)

        edge_index, edge_attr, all_pos, all_nt = self.process(pos)

        # Pack real kinematics
        Z = torch.zeros_like(all_pos)
        real = (all_nt == 0).squeeze(1)
        all_vel, all_pvel = Z.clone(), Z.clone()
        all_w, all_pw = Z.clone(), Z.clone()

        all_vel[real] = vel_t
        all_pvel[real] = vel_tm1
        all_w[real] = ang_vel_t
        all_pw[real] = ang_vel_tm1

        g = Data(edge_index=edge_index, edge_attr=edge_attr)
        g.node_feat = all_nt
        g.pos = all_pos
        g.vel = all_vel
        g.prev_vel = all_pvel
        g.ang_vel = all_w
        g.prev_ang_vel = all_pw
        g.y_dx = pos_tp1 - pos
        g.y_dv = vel_tp1 - vel_t
        g.y_dw = ang_vel_tp1 - ang_vel_t
        return g

    def get_omega(self, sim_time: float) -> torch.Tensor:
        """
        Defines the exact rotation profile of the cylinder over time.
        Ramps up to 12.56 rad/s, reverses down to -12.56 rad/s, and holds.
        """
        if 0.0 <= sim_time < 0.5:
            mag = 4.0 * sim_time
        elif 0.5 <= sim_time <= 1.5:
            mag = 4.0 - 4.0 * sim_time
        else:
            mag = -2.0  # Holds the peak reverse speed!

        # Convert Hz to radians/sec and orient along the rotation axis
        scalar = torch.tensor(2.0 * np.pi * mag, device=self.device, dtype=self.dtype)
        return scalar * self.axis

    @torch.no_grad()
    def rotating_walls(self, graph: Data, sim_time: float, sim_time_previous: float = None) -> Data:
        """
        Injects strictly physical rotational boundary conditions for both the
        curved walls and the flat end-caps of the rotating drum.
        """
        omega_cur = self.get_omega(sim_time)
        omega_prev = self.get_omega(sim_time_previous) if sim_time_previous is not None else omega_cur

        ghost = graph.node_feat.squeeze(1) == 1
        if not ghost.any():
            return graph

        # 1. Find the outward radial vector from the central axis to each ghost node
        gp = graph.pos[ghost]
        radial = (gp - self.center - ((gp - self.center) @ self.axis).unsqueeze(1) * self.axis)
        r_norm = radial.norm(dim=1, keepdim=True)
        r_hat = radial / (r_norm + 1e-9)

        # Curved wall ghosts have r_norm > R, so they clamp exactly to the surface R.
        # Flat cap ghosts have r_norm <= R, so they keep their true, slower local radius.
        surf_r = torch.clamp(r_norm, max=self.radius) * r_hat

        # 3. Calculate exact local tangential linear velocity: v = omega x r
        v_cur = torch.cross(omega_cur.expand_as(surf_r), surf_r, dim=1)
        v_prev = torch.cross(omega_prev.expand_as(surf_r), surf_r, dim=1)

        graph.vel[ghost] = v_cur
        graph.prev_vel[ghost] = v_prev

        # The boundary itself is a rotating solid body, so we impart its angular
        # velocity to the ghost nodes.
        graph.ang_vel[ghost] = omega_cur.expand_as(v_cur)
        graph.prev_ang_vel[ghost] = omega_prev.expand_as(v_cur)

        return graph


def insert_boundary(geo_data, boundary='cuboid', device='cpu', dtype=torch.float32):
    """
    Dispatcher to generate boundary initialization parameters for either
    a flat-walled cuboid or a cylindrical drum based on the provided geometry bounds.
    """
    (xmin, ymin, zmin), (xmax, ymax, zmax) = geo_data

    if boundary == 'cuboid':
        # Returns 6 plane equations [A, B, C, D]
        return {
            'front':  torch.tensor([ 1,  0,  0, -xmax], dtype=dtype, device=device),
            'back':   torch.tensor([-1,  0,  0,  xmin], dtype=dtype, device=device),
            'right':  torch.tensor([ 0,  1,  0, -ymax], dtype=dtype, device=device),
            'left':   torch.tensor([ 0, -1,  0,  ymin], dtype=dtype, device=device),
            'top':    torch.tensor([ 0,  0,  1, -zmax], dtype=dtype, device=device),
            'bottom': torch.tensor([ 0,  0, -1,  zmin], dtype=dtype, device=device)
        }

    elif boundary == 'cylinder':
        # Returns centroid, axis vector, radius, and length
        x_center = (xmax + xmin) / 2.0
        y_center = (ymax + ymin) / 2.0
        z_center = (zmax + zmin) / 2.0
        radius = (xmax - xmin) / 2.0
        height = zmax - zmin

        return {
                "axis": torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device),
                "radius": torch.tensor(radius, dtype=dtype, device=device),
                "center": torch.tensor([x_center, y_center, z_center], dtype=dtype, device=device),
                "length": torch.tensor(height, dtype=dtype, device=device),
            }
    else:
        raise ValueError(f"Unknown boundary type: {boundary}")
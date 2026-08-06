import torch
import torch.nn as nn
from utils.utils_dem import build_mlp_d
import torch.nn.functional as F


class RefFrameCalc(nn.Module):
    """
    Computes antisymmetric, local orthonormal basis vectors (a, b, c) for each edge.
    
    Ensures SE(3) invariance by constructing a coordinate system based on relative 
    positions and velocities. The basis (a, b, c) allows the network to learn 
    orientation-independent physical laws.
    """
    def __init__(self):
        super(RefFrameCalc, self).__init__()
        self.eps = 1e-8

    def forward(self, edge_index, 
                senders_pos, receivers_pos, 
                senders_vel, receivers_vel, 
                senders_prev_vel, receivers_prev_vel, 
                senders_omega, receivers_omega, 
                senders_prev_omega, receivers_prev_omega):
        
        # Calculate relative position (Edge vector)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True).clamp(min=self.eps)
        vector_a = rel_pos / dist

        # Preliminary vectors
        diff_vel = receivers_vel - senders_vel
        sum_vel = senders_vel + receivers_vel

        diff_prev_vel = receivers_prev_vel - senders_prev_vel
        sum_prev_vel = senders_prev_vel + receivers_prev_vel

        diff_omega = receivers_omega - senders_omega
        sum_omega = senders_omega + receivers_omega

        diff_prev_angvel = receivers_prev_omega - senders_prev_omega
        sum_prev_angvel = receivers_prev_omega + senders_prev_omega

        # Helper for normalizing
        def normalize(tensor):
            return tensor / tensor.norm(dim=1, keepdim=True).clamp(min=self.eps)

        b_i = normalize(torch.cross(diff_vel, vector_a, dim=1))
        b_ii = normalize(sum_vel)
        b_iii = normalize(torch.cross(diff_omega, vector_a, dim=1))
        b_iv = normalize(sum_omega)
        b_v = normalize(torch.cross(diff_prev_vel, vector_a, dim=1))
        b_vi = normalize(sum_prev_vel)
        b_vii = normalize(torch.cross(diff_prev_angvel, vector_a, dim=1))
        b_viii = normalize(sum_prev_angvel)

        # Combine to form b
        b = b_i + b_ii + b_iii + b_iv + b_v + b_vi + b_vii + b_viii

        # Gram-Schmidt like orthogonalization
        # Project b onto a
        b_prl_dot = (b * vector_a).sum(dim=1, keepdim=True)
        b_prl = b_prl_dot * vector_a
        b_prp = b - b_prl

        # vector_b is perpendicular to a
        vector_b = normalize(torch.cross(b_prp, vector_a, dim=1))
        
        # vector_c is perpendicular to a and b
        vector_c = normalize(torch.cross(b_prl, vector_b, dim=1))
   
        return vector_a, vector_b, vector_c


class NodeEncoder(nn.Module):
    """
    Encodes static scalar node properties (e.g., mass, friction) into a latent vector representation.
    """
    def __init__(self, node_in_f, latent_size, mlp_layers):
        super(NodeEncoder, self).__init__()
        self.node_encoder = build_mlp_d(node_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
    
    def forward(self, node_scalar_feat):
        return self.node_encoder(node_scalar_feat)


class InteractionEncoder(nn.Module):
    """
    Projects global state vectors onto the local reference frame for SE(3)-invariant message passing.
    
    Encodes the projected relative velocities and edge attributes into a latent interaction vector.
    """
    def __init__(self, edge_in_f, latent_size, mlp_layers):
        super(InteractionEncoder, self).__init__()
        # 12 features: v_t, v_tm1, w_t, w_tm1 projected onto 3 basis vectors (4 vectors * 3 dimensions = 12)
        self.edge_feat_encoder = build_mlp_d(12, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.edge_encoder = build_mlp_d(1, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3 * latent_size, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)

    def forward(self, edge_index, edge_dx_, # edge_attr, 
                vector_a, vector_b, vector_c,
                senders_v_t_, senders_v_tm1_, senders_w_t_, senders_w_tm1_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_, receivers_w_tm1_,
                node_latent):
        
        senders, receivers = edge_index

        # Stack basis vectors into a Rotation Matrix R = [a, b, c] of shape (E, 3, 3)
        basis = torch.stack([vector_a, vector_b, vector_c], dim=1) 

        def project(v):
            return torch.bmm(basis, v.unsqueeze(-1)).squeeze(-1)
        
        # Project senders
        s_vt_proj   = project(senders_v_t_)
        s_vtm1_proj = project(senders_v_tm1_)
        s_wt_proj   = project(senders_w_t_)
        s_wtm1_proj = project(senders_w_tm1_)

        # Project receivers (on the antisymmetric reference frame)
        r_vt_proj   = -project(receivers_v_t_)
        r_vtm1_proj = -project(receivers_v_tm1_)
        r_wt_proj   = -project(receivers_w_t_)
        r_wtm1_proj = -project(receivers_w_tm1_)

        # Concatenate features (12 features per node per edge)
        senders_features = torch.cat([s_vt_proj, s_vtm1_proj, s_wt_proj, s_wtm1_proj], dim=1)
        receivers_features = torch.cat([r_vt_proj, r_vtm1_proj, r_wt_proj, r_wtm1_proj], dim=1)
        
        # Edge encodings
        edge_dx_norm = edge_dx_.norm(dim=1, keepdim=True)

        # we only use the scalar edge features. In this case we only have the scalar dist (edge_attr is vector).
        edge_latent = self.edge_encoder(edge_dx_norm)

        # Encode features
        senders_latent = self.edge_feat_encoder(senders_features)
        receivers_latent = self.edge_feat_encoder(receivers_features)

        # Message Passing Input Construction
        node_sum = node_latent[senders] + node_latent[receivers]
        msg_input = torch.cat((senders_latent + receivers_latent, node_sum, edge_latent), dim=1)
        
        return self.interaction_encoder(msg_input)


class InteractionDecoder(torch.nn.Module):
    """
    Decodes interaction latent vectors into physical impulses (forces and torques).
    
    Enforces conservation of linear and angular momentum by decomposing impulses into 
    central and spin components, ensuring action-reaction symmetry.
    """
    def __init__(self, latent_size=128, mlp_layers=2):
        super(InteractionDecoder, self).__init__()
        self.i1_decoder          = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.i2_decoder          = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.lambdaij_scaler     = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.node_weight_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.eps = 1e-8

    def forward(self, edge_index, senders_pos, receivers_pos, vector_a, vector_b, vector_c, interaction_latent, node_latent):
        senders, receivers = edge_index
        
        # Decode coefficients 
        coeff_dp  = self.i1_decoder(interaction_latent)
        coeff_dl  = self.i2_decoder(interaction_latent)
        lambda_ij = self.lambdaij_scaler(interaction_latent)
        
        # Reconstruct change in momentum in global frame
        dpij = (
            coeff_dp[:, 0:1] * vector_a + 
            coeff_dp[:, 1:2] * vector_b + 
            coeff_dp[:, 2:3] * vector_c
        ) 
        # Reconstruct change in total angular momentum
        dlij = (
            coeff_dl[:, 0:1] * vector_a + 
            coeff_dl[:, 1:2] * vector_b + 
            coeff_dl[:, 2:3] * vector_c
        )

        # Node weights for reference point
        w_s = self.node_weight_decoder(node_latent[senders])
        w_r = self.node_weight_decoder(node_latent[receivers])
        
        # Weighted center r0ij
        denom = w_s + w_r + self.eps
        r0ij = (w_s * senders_pos + w_r * receivers_pos) / denom
        
        # Compute spin component
        dsij = dlij - torch.cross(receivers_pos - r0ij, dpij, dim=1)*lambda_ij
        
        return dpij, dsij


class Node_Internal_Dv_Decoder(torch.nn.Module):
    """
    Aggregates edge impulses and computes node state updates (dv, dw) using Newton's laws.
    """
    def __init__(self, latent_size=128, mlp_layers=2):
        super(Node_Internal_Dv_Decoder, self).__init__()
        self.m_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.i_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)

    def forward(self, edge_index, node_latent, fij, tij):
        senders, receivers = edge_index   
        num_nodes = node_latent.shape[0]
        
        # Decode physical properties
        m_inv = self.m_inv_decoder(node_latent)
        i_inv = self.i_inv_decoder(node_latent)
        
        # Aggregate Forces and Torques
        out_fij = node_latent.new_zeros((num_nodes, 3))
        out_tij = node_latent.new_zeros((num_nodes, 3))
        
        if edge_index.size(1) > 0:
            out_fij.index_add_(0, receivers, fij)
            out_tij.index_add_(0, receivers, tij)

        # Compute internal and angular impulse
        node_dv_int = m_inv * out_fij
        node_dw_int = i_inv * out_tij

        return node_dv_int, node_dw_int


class Scaler(torch.nn.Module):
    """
    Standardizes input features using the DEM training statistics dictionary.
    """
    def __init__(self):
        super(Scaler, self).__init__()
        self.eps = 1e-8

    def forward(self, senders_v_t, senders_v_tm1, receivers_v_t, receivers_v_tm1, 
                senders_w_t, senders_w_tm1, receivers_w_t, receivers_w_tm1, edge_dx, train_stats):
        
        # Extract max values directly from the dictionary generated by dataset.py
        v_max = train_stats['node_vel_max'].detach()
        w_max = train_stats['node_angvel_max'].detach()
        dx_norm_min = train_stats['edge_feat_min'].detach() 
        dx_norm_max = train_stats['edge_feat_max'].detach() 

        # Scale velocity features
        senders_v_t_ = senders_v_t / v_max
        senders_v_tm1_ = senders_v_tm1 / v_max
        receivers_v_t_ = receivers_v_t / v_max
        receivers_v_tm1_ = receivers_v_tm1 / v_max

        # Scale angular velocity features
        senders_w_t_ = senders_w_t / w_max
        senders_w_tm1_ = senders_w_tm1 / w_max
        receivers_w_t_ = receivers_w_t / w_max
        receivers_w_tm1_ = receivers_w_tm1 / w_max
        
        # Scale edge spatial differences
        norm_edge_dx = edge_dx.norm(dim=1, keepdim=True)
        scale_denom = dx_norm_max - dx_norm_min
        scaled_mag = (norm_edge_dx - dx_norm_min.detach()) / scale_denom
        edge_dx_ = scaled_mag * (edge_dx / norm_edge_dx)
        
        return senders_v_t_, senders_v_tm1_, receivers_v_t_, receivers_v_tm1_, senders_w_t_, senders_w_tm1_, receivers_w_t_, receivers_w_tm1_, edge_dx_

class Interaction_Block(torch.nn.Module):
    """
    Wrapper module combining interaction encoding, force/torque decoding, and node updates.
    """
    def __init__(self, edge_in_f, latent_size, mlp_layers):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(edge_in_f, latent_size, mlp_layers)
        self.interaction_decoder = InteractionDecoder(latent_size, mlp_layers)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size, mlp_layers)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_, vector_a, vector_b, vector_c, 
                s_vt_, s_vtm1_, s_wt_, s_wtm1_, 
                r_vt_, r_vtm1_, r_wt_, r_wtm1_,
                node_latent, residue=None, latent_history=False):
            
        if edge_index.size(1) == 0:
            interaction_latent = torch.zeros_like(node_latent)
            node_dv_int, node_dw_int = self.internal_dv_decoder(
                edge_index, node_latent, torch.zeros_like(node_latent[:, 0:3]), torch.zeros_like(node_latent[:, 0:3])
            )
            return node_dv_int, node_dw_int, interaction_latent

        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, # edge_attr,
            vector_a, vector_b, vector_c,
            s_vt_, s_vtm1_, s_wt_, s_wtm1_,
            r_vt_, r_vtm1_, r_wt_, r_wtm1_,
            node_latent
        )

        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)
        
        edge_force, edge_tau = self.interaction_decoder(
            edge_index, senders_pos, receivers_pos, 
            vector_a, vector_b, vector_c, 
            interaction_latent, node_latent
        )
        
        node_dv, node_dw = self.internal_dv_decoder(
            edge_index, node_latent, edge_force, edge_tau
        )
    
        return node_dv, node_dw, interaction_latent


class DynamicsSolver(torch.nn.Module):
    """
    Main physics simulation loop for DEM Rigid Body Dynamics.
    """
    def __init__(self, sample_step, train_stats, num_msgs=5, latent_size=128, mlp_layers=2, ext_force=False):
        super(DynamicsSolver, self).__init__()
        
        self.refframecalc = RefFrameCalc()       
        self.scaler = Scaler()                   
        
        # DEM Specific node and edge feature counts
        node_in_f = 1
        edge_in_f = 3
        self.node_encoder = NodeEncoder(node_in_f, latent_size, mlp_layers) 
        
        self.interaction_init_layer = Interaction_Block(edge_in_f, latent_size, mlp_layers)
        self.interaction_proc_layer = Interaction_Block(edge_in_f, latent_size, mlp_layers)

        # Optional external force decoder 
        if ext_force:
            self.ext_dv_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
                    
        self.num_messages = num_msgs
        self.sub_tstep = sample_step / num_msgs  
        self.train_stats = train_stats

    def forward(self, graph, interpretation=False):
        # 1. Unpack Graph Data
        pos = graph.pos.float()
        vel = graph.vel.float()
        prev_vel = graph.prev_vel.float()
        ang_vel = graph.ang_vel.float()
        prev_ang_vel = graph.prev_ang_vel.float()
        edge_attr = graph.edge_attr.float() # not used : not scalar
        node_type = graph.node_feat.float()
        
        # Mask for DEM spheres
        node_is_sphere = (node_type == 0).squeeze()
        
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index

        # 2. Initialize State Variables
        node_v_t = vel
        node_v_tm1 = prev_vel
        node_w_t = ang_vel
        node_w_tm1 = prev_ang_vel
        
        # 3. Initialize Accumulators
        sum_node_dv_ = torch.zeros_like(vel)
        sum_node_dw_ = torch.zeros_like(vel)
        sum_node_disp = torch.zeros_like(vel)
        
        # 4. Pre-compute Node Embeddings
        node_latent = self.node_encoder(node_type)
    

        # 5. Initialize Loop Variables
        current_pos = pos        
        residue = None 
        node_dv = torch.zeros_like(vel)
        node_dw = torch.zeros_like(vel)

        ext_dv = torch.zeros_like(vel)


        # Optional external force decoder (e.g., for gravity)
        if hasattr(self, 'ext_dv_decoder'):
            ext_dv[node_is_sphere] = self.ext_dv_decoder(node_latent)[node_is_sphere]

        # --- Message Passing & Integration Loop ---
        for i in range(self.num_messages):
            # Optional External Force Decoder

            if edge_index.size(1) > 0:
                # A. Gather Current State for Edges
                s_vt = node_v_t[senders]
                r_vt = node_v_t[receivers]
                s_vtm1 = node_v_tm1[senders]
                r_vtm1 = node_v_tm1[receivers]
                s_wt = node_w_t[senders]
                r_wt = node_w_t[receivers]
                s_wtm1 = node_w_tm1[senders]
                r_wtm1 = node_w_tm1[receivers]
                current_edge_dx = current_pos[receivers] - current_pos[senders]

                # B. Input Normalization
                s_vt_, s_vtm1_, r_vt_, r_vtm1_, s_wt_, s_wtm1_, r_wt_, r_wtm1_, edge_dx_ = self.scaler(
                    s_vt, s_vtm1, r_vt, r_vtm1, s_wt, s_wtm1, r_wt, r_wtm1, current_edge_dx, self.train_stats
                )

                # C. Frame Construction
                vec_a, vec_b, vec_c = self.refframecalc(
                    edge_index, current_pos[senders], current_pos[receivers],
                    s_vt, r_vt, s_vtm1, r_vtm1, s_wt, r_wt, s_wtm1, r_wtm1
                )

                # D. Interaction Block (Internal Forces)
                layer = self.interaction_init_layer if i == 0 else self.interaction_proc_layer
                history_flag = (i > 0)
                
                node_dv, node_dw, residue = layer(
                    edge_index, current_pos[senders], current_pos[receivers],
                    edge_dx_, vec_a, vec_b, vec_c,
                    s_vt_, s_vtm1_, s_wt_, s_wtm1_,
                    r_vt_, r_vtm1_, r_wt_, r_wtm1_, 
                    node_latent, residue=residue, latent_history=history_flag
                )
            else:
                node_dv = torch.zeros_like(node_v_t)
                node_dw = torch.zeros_like(node_w_t)

            # F. Symplectic Euler Integration (with DEM masks)
            # 1. Accumulate total velocity change
            node_dv[node_is_sphere] = node_dv[node_is_sphere] + ext_dv[node_is_sphere]
            sum_node_dv_[node_is_sphere] += node_dv[node_is_sphere]
            sum_node_dw_[node_is_sphere] += node_dw[node_is_sphere]
            
            # 2. Update Velocity
            node_vf = node_v_t.clone()
            node_vf[node_is_sphere] += node_dv[node_is_sphere]
            
            # 3. Update Angular Velocity
            node_wf = node_w_t.clone()
            node_wf[node_is_sphere] += node_dw[node_is_sphere]

            # 4. Calculate Displacement
            step_disp = torch.zeros_like(node_v_t)
            step_disp[node_is_sphere] = (node_v_t[node_is_sphere] + node_vf[node_is_sphere]) * (0.5 * self.sub_tstep)
            sum_node_disp[node_is_sphere] += step_disp[node_is_sphere]

            # 5. Update State for Next Iteration
            current_pos = current_pos + step_disp 
            
            node_v_tm1 = node_v_t.clone() 
            node_w_tm1 = node_w_t.clone() 
            
            node_v_t = node_vf
            node_w_t = node_wf                

        return sum_node_dv_, sum_node_dw_, sum_node_disp
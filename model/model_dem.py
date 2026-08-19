"""
Graph network for DEM rigid-body dynamics.

A contact graph is advanced over `num_msgs` sub-steps. Each sub-step builds a local frame on
every edge, encodes the contact into SE(3)-invariant scalars, decodes a pairwise impulse, and
integrates the spheres. Momentum is conserved by construction rather than by a penalty term.

See the accompanying paper for the formulation.
"""

# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE (EPFL),
# Switzerland, 2025.
# Authors: Vinay Sharma and Olga Fink, Laboratory of Intelligent Maintenance and
# Operations Systems (IMOS).
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils_dem import build_mlp_d


class RefFrameCalc(nn.Module):
    """Per-edge orthonormal frame (a, b, c), antisymmetric under sender/receiver exchange."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _normalize(self, t):
        # Clamped so a vanishing vector gives a finite result instead of NaN.
        return t / t.norm(dim=1, keepdim=True).clamp(min=self.eps)

    def forward(self,
                senders_pos, receivers_pos,
                senders_vel, receivers_vel,
                senders_prev_vel, receivers_prev_vel,
                senders_omega, receivers_omega,
                senders_prev_omega, receivers_prev_omega):
        n = self._normalize

        # First axis: along the edge.
        vector_a = n(receivers_pos - senders_pos)

        # Second axis seeded from the endpoint velocities, current and previous. Differences
        # go through a cross product with `a`, sums enter directly. Normalising each term
        # first stops one fast mode dominating the frame.
        b = (n(torch.cross(receivers_vel - senders_vel, vector_a, dim=1))
             + n(senders_vel + receivers_vel)
             + n(torch.cross(receivers_omega - senders_omega, vector_a, dim=1))
             + n(senders_omega + receivers_omega)
             + n(torch.cross(receivers_prev_vel - senders_prev_vel, vector_a, dim=1))
             + n(senders_prev_vel + receivers_prev_vel)
             + n(torch.cross(receivers_prev_omega - senders_prev_omega, vector_a, dim=1))
             + n(senders_prev_omega + receivers_prev_omega))

        # Gram-Schmidt against `a`.
        b_parallel = (b * vector_a).sum(dim=1, keepdim=True) * vector_a
        b_perp = b - b_parallel

        vector_b = n(torch.cross(b_perp, vector_a, dim=1))
        # From b_parallel rather than `a`: swapping the endpoints leaves b_parallel alone but
        # flips vector_b, so all three axes flip together and the frame stays antisymmetric.
        vector_c = n(torch.cross(b_parallel, vector_b, dim=1))

        return vector_a, vector_b, vector_c


class NodeEncoder(nn.Module):
    """Encodes the static node scalar (sphere vs boundary ghost) into a latent vector."""

    def __init__(self, node_in_f, latent_size, mlp_layers):
        super().__init__()
        self.node_encoder = build_mlp_d(node_in_f, latent_size, latent_size,
                                        num_layers=mlp_layers, lay_norm=True)

    def forward(self, node_scalar_feat):
        return self.node_encoder(node_scalar_feat)


class InteractionEncoder(nn.Module):
    """Builds the per-edge interaction latent from SE(3)-invariant scalars."""

    def __init__(self, latent_size, mlp_layers):
        super().__init__()
        # 12 = four vectors (v_t, v_t-1, w_t, w_t-1) x three frame components.
        self.edge_feat_encoder = build_mlp_d(12, latent_size, latent_size,
                                             num_layers=mlp_layers, lay_norm=True)
        self.edge_encoder = build_mlp_d(1, latent_size, latent_size,
                                        num_layers=mlp_layers, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3 * latent_size, latent_size, latent_size,
                                               num_layers=mlp_layers, lay_norm=True)

    def forward(self, edge_index, edge_dx_,
                vector_a, vector_b, vector_c,
                s_vt_, s_vtm1_, s_wt_, s_wtm1_,
                r_vt_, r_vtm1_, r_wt_, r_wtm1_,
                node_latent):
        senders, receivers = edge_index

        # Rows of `basis` are (a, b, c), so bmm(basis, v) gives (v.a, v.b, v.c).
        basis = torch.stack([vector_a, vector_b, vector_c], dim=1)

        def project(v):
            return torch.bmm(basis, v.unsqueeze(-1)).squeeze(-1)

        sender_feats = torch.cat(
            [project(s_vt_), project(s_vtm1_), project(s_wt_), project(s_wtm1_)], dim=1)
        # Negated so that, with the frame flipping under exchange, both half-edges of a
        # contact give the same latent.
        receiver_feats = -torch.cat(
            [project(r_vt_), project(r_vtm1_), project(r_wt_), project(r_wtm1_)], dim=1)

        msg_input = torch.cat((
            # Sums, not concatenation, so the message is order-independent.
            self.edge_feat_encoder(sender_feats) + self.edge_feat_encoder(receiver_feats),
            node_latent[senders] + node_latent[receivers],
            self.edge_encoder(edge_dx_.norm(dim=1, keepdim=True)),
        ), dim=1)

        return self.interaction_encoder(msg_input)


class InteractionDecoder(nn.Module):
    """
    Decodes the interaction latent into the pairwise impulses.

    Reads a linear momentum exchange dp_ij and a total angular momentum exchange dl_ij as
    coefficients in the edge frame, then returns the spin delivered to the receiver,

        ds_ij = dl_ij - (x_recv - r0_ij) x dp_ij

    about the learned weighted centre r0_ij.
    """

    def __init__(self, latent_size=128, mlp_layers=2):
        super().__init__()
        self.dp_decoder = build_mlp_d(latent_size, latent_size, 3,
                                      num_layers=mlp_layers, lay_norm=False)
        self.dl_decoder = build_mlp_d(latent_size, latent_size, 3,
                                      num_layers=mlp_layers, lay_norm=False)
        self.node_weight_decoder = build_mlp_d(latent_size, latent_size, 1,
                                               num_layers=mlp_layers, lay_norm=False)
        self.eps = 1e-8

    def forward(self, edge_index, node_latent,
                senders_pos, receivers_pos,
                vector_a, vector_b, vector_c, interaction_latent):

        # The heads emit frame coefficients; rebuild the world-frame vectors from them.
        def to_global(c):
            return c[:, 0:1] * vector_a + c[:, 1:2] * vector_b + c[:, 2:3] * vector_c

        dpij = to_global(self.dp_decoder(interaction_latent))
        dlij = to_global(self.dl_decoder(interaction_latent))

        senders, receivers = edge_index

        # One shared MLP for both endpoints keeps r0ij symmetric under exchange, so both
        # half-edges subtract the orbital part about the same point.
        senders_weight = self.node_weight_decoder(node_latent[senders])
        receivers_weight = self.node_weight_decoder(node_latent[receivers])

        r0ij = ((senders_weight * senders_pos + receivers_weight * receivers_pos)
                / (senders_weight + receivers_weight + self.eps))

        # Spin is the total minus the orbital part. The cross term's coefficient must stay
        # exactly one or the pair contributions stop cancelling.
        dsij = dlij - torch.cross(receivers_pos - r0ij, dpij, dim=1)
        return dpij, dsij


class NodeImpulseAggregator(nn.Module):
    """Newton's laws at the node: dv = m^-1 * sum(dp), dw = I^-1 * sum(ds)."""

    def forward(self, edge_index, inv_mass, inv_inertia, num_nodes, dpij, dsij):
        receivers = edge_index[1]

        # Scatter-add every incoming edge contribution onto its receiver.
        summed_dp = dpij.new_zeros((num_nodes, 3))
        summed_ds = dsij.new_zeros((num_nodes, 3))
        summed_dp.index_add_(0, receivers, dpij)
        summed_ds.index_add_(0, receivers, dsij)

        return inv_mass * summed_dp, inv_inertia * summed_ds


class Scaler(nn.Module):
    """Normalises velocities and the edge displacement with the training statistics."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def forward(self, s_vt, s_vtm1, r_vt, r_vtm1,
                s_wt, s_wtm1, r_wt, r_wtm1, edge_dx, train_stats):
        v_max = train_stats['node_vel_max'].detach() + self.eps
        w_max = train_stats['node_angvel_max'].detach() + self.eps
        dx_min = train_stats['edge_feat_min'].detach()
        dx_max = train_stats['edge_feat_max'].detach()

        # Min-max the separation magnitude and reattach it to the unit direction, keeping
        # overlap depth well resolved near contact. Directions are left untouched throughout.
        norm_dx = edge_dx.norm(dim=1, keepdim=True).clamp(min=self.eps)
        scaled_mag = (norm_dx - dx_min) / (dx_max - dx_min).clamp(min=self.eps)
        edge_dx_ = scaled_mag * (edge_dx / norm_dx)

        return (s_vt / v_max, s_vtm1 / v_max, r_vt / v_max, r_vtm1 / v_max,
                s_wt / w_max, s_wtm1 / w_max, r_wt / w_max, r_wtm1 / w_max,
                edge_dx_)


class InteractionBlock(nn.Module):
    """One message-passing pass: encode the contact, decode the impulses, apply them."""

    def __init__(self, latent_size, mlp_layers):
        super().__init__()
        self.inv_mass_decoder = build_mlp_d(latent_size, latent_size, 1,
                                            num_layers=mlp_layers, lay_norm=False)
        self.inv_inertia_decoder = build_mlp_d(latent_size, latent_size, 1,
                                               num_layers=mlp_layers, lay_norm=False)
        self.interaction_encoder = InteractionEncoder(latent_size, mlp_layers)
        self.interaction_decoder = InteractionDecoder(latent_size, mlp_layers)
        self.impulse_aggregator = NodeImpulseAggregator()
        self.layer_norm = nn.LayerNorm(latent_size)
        self.eps = 1e-8

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_,
                vector_a, vector_b, vector_c,
                s_vt_, s_vtm1_, s_wt_, s_wtm1_,
                r_vt_, r_vtm1_, r_wt_, r_wtm1_,
                node_latent, residue=None, latent_history=False):
        # Softplus keeps both positive. Unconstrained they can go negative, which turns the
        # angular update into anti-damping.
        inv_mass = F.softplus(self.inv_mass_decoder(node_latent)) + self.eps
        inv_inertia = F.softplus(self.inv_inertia_decoder(node_latent)) + self.eps

        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, vector_a, vector_b, vector_c,
            s_vt_, s_vtm1_, s_wt_, s_wtm1_, r_vt_, r_vtm1_, r_wt_, r_wtm1_, node_latent)

        # From the second sub-step on, carry the previous latent forward as a residual.
        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)

        dpij, dsij = self.interaction_decoder(
            edge_index, node_latent,
            senders_pos, receivers_pos,
            vector_a, vector_b, vector_c, interaction_latent)

        node_dv, node_dw = self.impulse_aggregator(
            edge_index, inv_mass, inv_inertia, node_latent.size(0), dpij, dsij)

        return node_dv, node_dw, interaction_latent


class DynamicsSolver(nn.Module):
    """
    DEM simulation loop.

    Each of `num_msgs` sub-steps of length sample_step / num_msgs:
      1. gather the per-edge endpoint states and normalise them,
      2. build the antisymmetric edge frames,
      3. run an interaction block to obtain dv and dw,
      4. add the optional external dv and integrate.

    Returns the summed dv, dw and displacement over the whole step.
    """

    def __init__(self, sample_step, train_stats, num_msgs=5,
                 latent_size=128, mlp_layers=2, ext_force=False):
        super().__init__()
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(1, latent_size, mlp_layers)

        # A dedicated block for the first sub-step, then one shared block for the rest.
        self.interaction_init_layer = InteractionBlock(latent_size, mlp_layers)
        self.interaction_proc_layer = InteractionBlock(latent_size, mlp_layers)

        if ext_force:
            # The external field acts along the y axis; only its signed magnitude is learned,
            # so it cannot pick up the x and z components the data does not have. Deliberately
            # not rotation-equivariant: it stands for a fixed field like gravity.
            axis = torch.as_tensor((0, 1, 0), dtype=torch.float32)
            self.register_buffer("ext_axis", axis / axis.norm())
            self.ext_dv_decoder = build_mlp_d(latent_size, latent_size, 1,
                                              num_layers=mlp_layers, lay_norm=False)

        self.num_messages = num_msgs
        self.sub_tstep = sample_step / num_msgs
        self.train_stats = train_stats

    def forward(self, graph):
        pos = graph.pos.float()
        node_v_t = graph.vel.float()
        node_v_tm1 = graph.prev_vel.float()
        node_w_t = graph.ang_vel.float()
        node_w_tm1 = graph.prev_ang_vel.float()
        node_type = graph.node_feat.float()

        edge_index = graph.edge_index.long()
        senders, receivers = edge_index
        has_edges = edge_index.size(1) > 0

        # Ghosts are held fixed within the step; only spheres integrate. They are images, not
        # bodies, so integrating them would let the impulse push them around as free particles.
        # The rollout rebuilds them by reflection again at the next step.
        is_sphere = (node_type == 0).squeeze(-1).unsqueeze(-1)

        node_latent = self.node_encoder(node_type)

        # Decoded once, reapplied each sub-step, so its contribution scales with num_msgs.
        # Only a magnitude is decoded; the direction is the fixed axis.
        ext_acc = torch.zeros_like(node_v_t)
        if hasattr(self, 'ext_dv_decoder'):
            ext_acc = torch.where(is_sphere,
                                 self.ext_dv_decoder(node_latent) * self.ext_axis, ext_acc)

        sum_node_dv = torch.zeros_like(node_v_t)
        sum_node_dw = torch.zeros_like(node_v_t)
        sum_node_disp = torch.zeros_like(node_v_t)

        current_pos = pos
        residue = None

        for i in range(self.num_messages):
            if has_edges:
                s_vt, r_vt = node_v_t[senders], node_v_t[receivers]
                s_vtm1, r_vtm1 = node_v_tm1[senders], node_v_tm1[receivers]
                s_wt, r_wt = node_w_t[senders], node_w_t[receivers]
                s_wtm1, r_wtm1 = node_w_tm1[senders], node_w_tm1[receivers]
                edge_dx = current_pos[receivers] - current_pos[senders]

                (s_vt_, s_vtm1_, r_vt_, r_vtm1_,
                 s_wt_, s_wtm1_, r_wt_, r_wtm1_, edge_dx_) = self.scaler(
                    s_vt, s_vtm1, r_vt, r_vtm1, s_wt, s_wtm1, r_wt, r_wtm1,
                    edge_dx, self.train_stats)

                # Unscaled states: each term is normalised anyway, so scaling would cancel.
                vec_a, vec_b, vec_c = self.refframecalc(
                    current_pos[senders], current_pos[receivers],
                    s_vt, r_vt, s_vtm1, r_vtm1, s_wt, r_wt, s_wtm1, r_wtm1)

                layer = self.interaction_init_layer if i == 0 else self.interaction_proc_layer
                node_dv, node_dw, residue = layer(
                    edge_index, current_pos[senders], current_pos[receivers], edge_dx_,
                    vec_a, vec_b, vec_c,
                    s_vt_, s_vtm1_, s_wt_, s_wtm1_, r_vt_, r_vtm1_, r_wt_, r_wtm1_,
                    node_latent, residue=residue, latent_history=(i > 0))
            else:
                # No contacts this sub-step, so nothing accelerates.
                node_dv = torch.zeros_like(node_v_t)
                node_dw = torch.zeros_like(node_w_t)

            zero = torch.zeros_like(node_dv)
            node_dv = torch.where(is_sphere, node_dv + ext_acc*self.sub_tstep, zero)
            node_dw = torch.where(is_sphere, node_dw, zero)

            node_v_next = node_v_t + node_dv
            node_w_next = node_w_t + node_dw
            # Trapezoidal rule: average the velocities across the sub-step.
            step_disp = torch.where(
                is_sphere, 0.5 * (node_v_t + node_v_next) * self.sub_tstep, zero)

            sum_node_dv = sum_node_dv + node_dv
            sum_node_dw = sum_node_dw + node_dw
            sum_node_disp = sum_node_disp + step_disp

            # Advance the state; the current values become the previous ones.
            current_pos = current_pos + step_disp
            node_v_tm1, node_w_tm1 = node_v_t, node_w_t
            node_v_t, node_w_t = node_v_next, node_w_next

        return sum_node_dv, sum_node_dw, sum_node_disp

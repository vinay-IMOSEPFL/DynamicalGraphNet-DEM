# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE (EPFL),
# Switzerland, 2025.
# Authors: Vinay Sharma and Olga Fink, Laboratory of Intelligent Maintenance and
# Operations Systems (IMOS).
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import torch
import torch.nn as nn

def build_mlp_d(in_size, hidden_size, out_size, num_layers=1, lay_norm=True, use_sigmoid=False, use_softmax=False):
    """Exact MLP builder."""
    if use_sigmoid and use_softmax:
        raise ValueError("Only one of use_sigmoid or use_softmax can be true.")

    layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]

    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(hidden_size, out_size))
    module = nn.Sequential(*layers)

    if lay_norm:
        module = nn.Sequential(module, nn.LayerNorm(normalized_shape=out_size))

    if use_sigmoid:
        module = nn.Sequential(module, nn.Sigmoid())

    if use_softmax:
        module = nn.Sequential(module, nn.Softmax(dim=-1))

    return module


def calculate_linear_momentum(node_vel):
    return torch.sum(node_vel, dim=0)


def calculate_angular_momentum(node_pos, node_vel, node_angvel, radius=0.005/2.0):
    mi_sphere_per_unit_mass = (2/5) * radius**2
    angular_momentum_per_unit_mass = torch.zeros_like(node_angvel[0])
    for i in range(node_pos.shape[0]):
        r = node_pos[i]
        p = node_vel[i]
        w = node_angvel[i]
        L_linear = torch.cross(r, p)
        L_angular = mi_sphere_per_unit_mass * w
        angular_momentum_per_unit_mass += L_linear + L_angular

    return angular_momentum_per_unit_mass


def calculate_energy(node_vel, node_angvel, radius=0.005/2.0):
    mi_sphere_per_unit_mass = (2/5) * radius**2
    translational_ke = 0.5 * torch.sum(node_vel * node_vel, dim=1)
    rotational_ke = 0.5 * mi_sphere_per_unit_mass * torch.sum(node_angvel * node_angvel, dim=1)
    total_ke_per_unit_mass = translational_ke + rotational_ke
    return torch.sum(total_ke_per_unit_mass)
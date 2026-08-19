# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, boundaries, threshold, train_stats, time_step):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_history = []
        self.cur_dir = os.getcwd()
        self.model_dir = os.path.join(self.cur_dir, 'saved_models')
        self.boundaries = boundaries
        self.threshold = threshold
        self.train_stats = train_stats

        # Safely detach stats so no gradients track through them
        self.mean_node_dx = train_stats['dx_mean'].to(device).detach()
        self.std_node_dx = train_stats['dx_std'].to(device).detach()

        self.mean_node_dv = train_stats['dv_mean'].to(device).detach()
        self.std_node_dv = train_stats['dv_std'].to(device).detach()

        self.mean_node_dw = train_stats['dw_mean'].to(device).detach()
        self.std_node_dw = train_stats['dw_std'].to(device).detach()

        self.time_step = time_step
        self.loss = 0.0

    def train(self, ingraph, accumulate=False, accumulation_steps=1):
        """
        Executes a single forward/backward pass with support for gradient accumulation and clipping.
        """
        self.model.train()

        pred_node_dvel, pred_node_dangvel, pred_node_disp = self.model(ingraph)

        node_type = ingraph.node_feat
        node_is_wall = (node_type == 1).any(dim=1)
        node_is_sphere = ~node_is_wall

        actual_node_dvel = ingraph.y_dv.float()
        actual_node_dangvel = ingraph.y_dw.float()
        actual_node_disp = ingraph.y_dx.float()

        # Standardize outputs
        pred_node_dvel_ = (pred_node_dvel - self.mean_node_dv.float()) / self.std_node_dv.float()
        pred_node_dangvel_ = (pred_node_dangvel - self.mean_node_dw.float()) / self.std_node_dw.float()
        pred_node_disp_ = (pred_node_disp - self.mean_node_dx.float()) / self.std_node_dx.float()

        # Standardize targets
        actual_node_dvel_ = (actual_node_dvel - self.mean_node_dv.float()) / self.std_node_dv.float()
        actual_node_dangvel_ = (actual_node_dangvel - self.mean_node_dw.float()) / self.std_node_dw.float()
        actual_node_disp_ = (actual_node_disp - self.mean_node_dx.float()) / self.std_node_dx.float()

        # Mask the loss so it is only calculated on active spheres (not boundaries)
        loss1 = F.mse_loss(pred_node_dvel_[node_is_sphere], actual_node_dvel_)
        loss2 = F.mse_loss(pred_node_dangvel_[node_is_sphere], actual_node_dangvel_)
        loss3 = F.mse_loss(pred_node_disp_[node_is_sphere], actual_node_disp_)

        # Displacement, velocity and angular velocity contribute equally.
        raw_loss = loss1 + loss2 + loss3

        # Scale so the accumulated sum averages over the effective batch.
        scaled_loss = raw_loss / accumulation_steps
        scaled_loss.backward()

        # Only step the optimizer when the accumulation cycle finishes
        if not accumulate:
            # Gradient clipping to prevent exploding gradients during volatile collisions
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.loss = raw_loss.item()
        self.train_history.append(self.loss)

    def save_model(self, tag):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        path = os.path.join(self.model_dir, f'model_checkpoint_{tag}.pth')
        torch.save(self.model.state_dict(), path)

def train_one_epoch(trainer, loader, pbar_desc="", smooth=0.98, accumulation_steps=1):
    """
    Train for one epoch with an integrated gradient accumulation loop.
    You can alter `accumulation_steps` to mimic larger batch sizes.
    """
    avg, total_loss, count = None, 0.0, 0

    # Zero gradients at the very beginning of the epoch
    trainer.optimizer.zero_grad()

    with tqdm(loader, desc=pbar_desc, leave=False, mininterval=0.25) as bar:
        for i, batch in enumerate(bar):
            if batch.edge_index.numel() == 0:
                continue

            batch = batch.to(trainer.device)

            # Determine if we should accumulate or execute a step
            is_last_batch = (i == len(loader) - 1)
            accumulate = ((i + 1) % accumulation_steps != 0) and not is_last_batch

            trainer.train(batch, accumulate=accumulate, accumulation_steps=accumulation_steps)
            loss = trainer.loss

            total_loss += loss
            count += 1
            avg = loss if avg is None else (smooth * avg + (1-smooth) * loss)

            bar.set_postfix({
                "batch": f"{loss:8.3e}",
                "avg":   f"{avg:8.3e}"
            })

    return total_loss / count if count else 0.0
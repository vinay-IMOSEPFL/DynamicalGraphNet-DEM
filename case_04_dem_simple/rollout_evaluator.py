"""
Autoregressive rollout evaluation for the homogeneous DEM cases.

The model consumes its own predictions for the full length of the rollout, with no
ground truth injected after the initial state. Ground truth is read only to measure
error, so the reported values reflect accumulated drift rather than one-step accuracy.
"""

# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import shutil
import torch
import concurrent.futures

def _render_frame_worker(args):
    """
    Top-level helper function to render a single frame in a background CPU process.
    Top-level definition is required for Python's multiprocessing pickling.
    """
    file_path, plot_folder_path, f_idx, total_steps, plot_region, bottom_wall, experiment_name = args

    print(f"Rendering rollout step {f_idx + 1} / {total_steps}...")
    data = torch.load(file_path, map_location='cpu')

    from case_04_dem_simple.visualization import plot_snapshot_test
    plot_snapshot_test(
        data['pred_pos'].numpy(), data['pred_angpos'].numpy(),
        data['gt_pos'].numpy(), data['gt_angpos'].numpy(),
        data['pred_vel'].numpy(), data['pred_angvel'].numpy(),
        data['gt_vel'].numpy(), data['gt_angvel'].numpy(),
        data['mae_pos'].item(), data['mae_vel'].item(), data['mae_angvel'].item(), f_idx,
        plot_folder_path, experiment_name, plot_region=plot_region, bottom_wall=bottom_wall
    )


def evaluate_rollout(test_loader, model, interaction, device, train_stats, time_step=1e-4,
                     start=0, end=200, plot=False, save_data=False, frequency=1,
                     experiment_name='Val', save_folder='.',
                     plot_region=(-0.03, 0.03), bottom_wall=False):
    """
    Performs an autoregressive rollout evaluation of the trained dynamics model.

    Instead of using ground truth data at every step (1-step prediction), this function
    feeds the model's own predictions back into itself for 'end - start' time steps.
    This measures the model's stability and error accumulation over long time horizons.

    Args:
        test_loader (DataLoader): PyTorch Geometric DataLoader providing the sequential graph data.
        model (nn.Module): The trained Graph Neural Network dynamics solver.
        interaction (object): Boundary interaction handler (e.g., SphereWallInteraction) to
                              update graph topology as particles move.
        device (torch.device): Computation device.
        train_stats (dict): Dictionary of maximum absolute values used for scaling error metrics.
        time_step (float): Physical time duration of a single simulation step.
        start (int): Time step index to begin the rollout.
        end (int): Time step index to end the rollout.
        plot (bool): If True, generates and saves 3D scatter plots comparing predictions to ground truth.
        save_data (bool): If True, serializes the kinematic states to disk at each step.
        frequency (int): Frame interval for saving 3D plots (e.g., 10 means plot every 10th frame).
        experiment_name (str): Identifier used to name the output directories.
        save_folder (str): Root directory where output folders will be created.
        plot_region (tuple): Axis limits for the 3D plots.
        bottom_wall (bool): If True, renders a visual floor in the 3D plots.

    Returns:
        tuple: (lst_error_pos, lst_error_vel, lst_error_angvel, predicted_traj_sys, grndtruth_traj_sys)
               Containing the scaled MAE sequences and the full kinematic history for downstream analysis.
    """
    model.eval()

    # Extract normalization factors to compute dimensionless error metrics
    eps = 1e-8
    pos_scale = train_stats['edge_feat_max'].item() + eps
    vel_scale = train_stats['node_vel_max'].item() + eps
    angvel_scale = train_stats['node_angvel_max'].item() + eps

    with torch.no_grad():
        lst_error_pos, lst_error_vel, lst_error_angvel = [], [], []
        predicted_traj_sys, grndtruth_traj_sys = [], []

        # Prepare output directories
        plot_folder_path = os.path.join(save_folder, experiment_name, 'rollout_plots')

        # Store data universally in a tmp folder first.
        data_folder_path = os.path.join(save_folder, experiment_name, 'tmp_rollout_data')
        if os.path.exists(data_folder_path): shutil.rmtree(data_folder_path)
        os.makedirs(data_folder_path, exist_ok=True)

        if plot:
            if os.path.exists(plot_folder_path): shutil.rmtree(plot_folder_path)
            os.makedirs(plot_folder_path, exist_ok=True)

        ingraph = None
        total_steps = end - start

        # 1. Autoregressive rollout.
        for i, graph_t0 in enumerate(test_loader):
            if i > end or i >= len(test_loader): break
            if i < start: continue

            step_idx = i - start
            print(f"Evaluating rollout step {step_idx + 1} / {total_steps}...")

            graph_t0 = graph_t0.to(device)

            # Initialize the rollout state on the very first requested frame
            if i == start:
                ingraph = graph_t0.detach().clone()
                # Initialize visual trackers for angular position (orientation)
                groundtruth_node_angpos = torch.zeros_like(graph_t0.pos)
                groundtruth_node_angpos[:, -1:] += 1
                node_angpos_0 = torch.zeros_like(graph_t0.pos)
                node_angpos_0[:, -1:] += 1

            # 1. Forward Pass: Predict the change in velocity, angular velocity, and position
            dv_pred, dw_pred, dx_pred = model(ingraph.detach())

            # Identify the real particles (ignoring ghost/boundary nodes)
            sphere_node = (ingraph.node_feat == 0).any(dim=1)

            # 2. Kinematic Integration (Euler Step)
            node_vel_0 = ingraph.vel
            node_vel_1 = torch.zeros_like(ingraph.vel)
            node_vel_1[sphere_node] = node_vel_0[sphere_node] + dv_pred[sphere_node].detach()

            node_angvel_0 = ingraph.ang_vel
            node_angvel_1 = torch.zeros_like(ingraph.vel)
            node_angvel_1[sphere_node] = node_angvel_0[sphere_node] + dw_pred[sphere_node].detach()

            node_pos_0 = ingraph.pos
            node_pos_1 = node_pos_0 + dx_pred.detach()

            # Update angular positions (orientations) strictly for visualization purposes
            node_angpos_1 = node_angpos_0 + 0.5 * (node_angvel_1 + node_angvel_0) * time_step
            node_angpos_0 = node_angpos_1

            # Extract the integrated states strictly for the real spheres
            pred_pos_sphere = node_pos_1[sphere_node]
            pred_vel_sphere = node_vel_1[sphere_node]
            pred_angvel_sphere = node_angvel_1[sphere_node]
            pred_angpos_sphere = node_angpos_1[sphere_node]

            # 3. Dynamic Topology Update
            edge_index, edge_attr, pos_updated, _ = interaction.process(pred_pos_sphere.detach())

            ingraph.edge_index = edge_index.detach().to(device)
            ingraph.edge_attr = edge_attr.detach().to(device)
            ingraph.pos = pos_updated.detach().to(device)
            ingraph.vel = node_vel_1.detach().to(device)
            ingraph.prev_vel = node_vel_0.detach().to(device)
            ingraph.ang_vel = node_angvel_1.detach().to(device)
            ingraph.prev_ang_vel = node_angvel_0.detach().to(device)
            ingraph.node_feat = ingraph.node_feat.detach().to(device)

            # 4. Ground Truth Tracking
            ang_disp = torch.zeros_like(graph_t0.ang_vel)
            ang_disp[sphere_node] = (2 * graph_t0.ang_vel[sphere_node] + graph_t0.y_dw) * 0.5 * time_step
            groundtruth_node_angpos = groundtruth_node_angpos + ang_disp

            groundtruth_node_pos_particles = graph_t0.pos[sphere_node] + graph_t0.y_dx
            groundtruth_node_vel_particles = graph_t0.vel[sphere_node] + graph_t0.y_dv
            groundtruth_node_angvel_particles = graph_t0.ang_vel[sphere_node] + graph_t0.y_dw
            groundtruth_node_angpos_particles = groundtruth_node_angpos[sphere_node]

            # 5. Error Calculation
            error_pos = torch.mean(torch.abs((pred_pos_sphere - groundtruth_node_pos_particles) / pos_scale))
            error_vel = torch.mean(torch.abs((pred_vel_sphere - groundtruth_node_vel_particles) / vel_scale))
            error_angvel = torch.mean(torch.abs((pred_angvel_sphere - groundtruth_node_angvel_particles) / angvel_scale))

            lst_error_pos.append(error_pos.item())
            lst_error_vel.append(error_vel.item())
            lst_error_angvel.append(error_angvel.item())

            # 6. Universal Caching (Replaces memory dictionary with disk flush)
            snapshot_data = {
                'pred_pos': pred_pos_sphere.cpu(),
                'pred_vel': pred_vel_sphere.cpu(),
                'pred_angvel': pred_angvel_sphere.cpu(),
                'pred_angpos': pred_angpos_sphere.cpu(),
                'gt_pos': groundtruth_node_pos_particles.cpu(),
                'gt_vel': groundtruth_node_vel_particles.cpu(),
                'gt_angvel': groundtruth_node_angvel_particles.cpu(),
                'gt_angpos': groundtruth_node_angpos_particles.cpu(),
                'mae_pos': error_pos.cpu(),
                'mae_vel': error_vel.cpu(),
                'mae_angvel': error_angvel.cpu()
            }
            torch.save(snapshot_data, os.path.join(data_folder_path, f"snapshot_{i:04d}.pt"))

            if not save_data:
                # Keep a running list of full system states to render global physics conservation plots later
                predicted_traj_sys.append(torch.hstack((
                    pred_pos_sphere, pred_vel_sphere, pred_angvel_sphere, pred_angpos_sphere
                )))
                grndtruth_traj_sys.append(torch.hstack((
                    groundtruth_node_pos_particles, groundtruth_node_vel_particles,
                    groundtruth_node_angvel_particles, groundtruth_node_angpos_particles
                )))

        # 2. Render frames from the snapshots, one process per core.
        if plot:
            print(f"\nPhysics rollout complete. Rendering frames to {plot_folder_path} using multiprocessing...")
            pt_files = sorted([os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pt')])

            # Prepare the tasks list (filters by frequency to avoid spinning up unnecessary processes)
            tasks = []
            for file_path in pt_files:
                f_idx = int(os.path.basename(file_path).split('_')[1].split('.')[0])
                if f_idx % frequency == 0 or f_idx == 0:
                    tasks.append((file_path, plot_folder_path, f_idx, total_steps, plot_region, bottom_wall, experiment_name))

            # Map the rendering helper across all available CPU cores
            with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                for _ in executor.map(_render_frame_worker, tasks):
                    pass

        # 3. Rebuild the trajectory lists from disk.
        # When save_data is enabled the per-step states were flushed to disk rather than
        # held in RAM, so the in-memory trajectory lists are still empty. Callers
        # (e.g. plot_physics_panel) need them, so rebuild them from the snapshots here.
        if save_data:
            for f in sorted(os.listdir(data_folder_path)):
                if not f.endswith('.pt'):
                    continue
                snap = torch.load(os.path.join(data_folder_path, f), map_location='cpu')
                predicted_traj_sys.append(torch.hstack((
                    snap['pred_pos'], snap['pred_vel'], snap['pred_angvel'], snap['pred_angpos']
                )))
                grndtruth_traj_sys.append(torch.hstack((
                    snap['gt_pos'], snap['gt_vel'], snap['gt_angvel'], snap['gt_angpos']
                )))

        # 4. Keep or discard the snapshot directory.
        if save_data:
            final_data_folder = os.path.join(save_folder, experiment_name, 'rollout_data')
            if os.path.exists(final_data_folder):
                shutil.rmtree(final_data_folder)
            os.rename(data_folder_path, final_data_folder)
        else:
            shutil.rmtree(data_folder_path, ignore_errors=True)

        return lst_error_pos, lst_error_vel, lst_error_angvel, predicted_traj_sys, grndtruth_traj_sys
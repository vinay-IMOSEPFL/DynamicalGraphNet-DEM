"""
Autoregressive rollout evaluation for the heterogeneous DEM cases.

The model consumes its own predictions for the full length of the rollout, with no
ground truth injected after the initial state. Ground truth is read only to measure
error, so the reported values reflect accumulated drift rather than one-step accuracy.
"""

import os
import shutil
import torch
import concurrent.futures

def _render_frame_worker(args):
    """
    Top-level helper function to render a single frame in a background process.
    Top-level definition is required for Python's multiprocessing pickling.
    """
    file_path, plot_folder_path, f_idx, total_steps, is_cylinder, bounds_dict, plot_region, bottom_wall, experiment_name = args

    print(f"Rendering rollout step {f_idx + 1} / {total_steps}...")
    data = torch.load(file_path, map_location='cpu')

    if is_cylinder:
        from case_05_dem_hard.visualization import plot_snapshot_cyl
        plot_snapshot_cyl(
            data['pred_pos'].numpy(), data['gt_pos'].numpy(),
            sim_time=data['t_model'], frame_idx=f_idx,
            error_pos=data['mae_pos'].item(), error_vel=data['mae_vel'].item(), error_angvel=data['mae_ang'].item(),
            plot_folder_path=plot_folder_path, boundaries=bounds_dict, model_name='Dynami-CAL'
        )
    else:
        from case_05_dem_hard.visualization import plot_snapshot_test
        plot_snapshot_test(
            data['pred_pos'].numpy(), data['pred_angpos'].numpy(), data['gt_pos'].numpy(), data['gt_angpos'].numpy(),
            data['pred_vel'].numpy(), data['pred_ang'].numpy(), data['gt_vel'].numpy(), data['gt_ang'].numpy(),
            data['mae_pos'].item(), data['mae_vel'].item(), data['mae_ang'].item(), f_idx,
            plot_folder_path, experiment_name, plot_region=plot_region, bottom_wall=bottom_wall
        )

def evaluate_rollout(test_loader, model, interaction, device, train_stats,
                     dt_model=1e-4, dt_loader=1e-3, microsteps_per_sync=10,
                     start=0, end=200, plot=False, save_data=False, frequency=1,
                     experiment_name='Val', save_folder='.',
                     plot_region=(-0.03, 0.03), bottom_wall=False):
    """
    Run an autoregressive rollout and return per-step scaled errors and trajectories.

    Dual timestep
    -------------
    The cylinder ground truth was saved every `dt_loader` = 1e-3 s to keep the dataset
    to a manageable size, but the model must integrate at `dt_model` = 1e-4 s to resolve
    contact events. The rollout therefore takes `microsteps_per_sync` internal steps
    between consecutive ground truth frames, comparing states only where the two clocks
    coincide. For the cuboid cases the two timesteps are equal and this reduces to one
    model step per frame.

    Returns:
        (pos_err, vel_err, angvel_err, predicted_traj, groundtruth_traj) - the three error
        sequences are dimensionless, scaled by the training-set maxima in `train_stats`.
    """
    model.eval()

    # Extract normalization factors to compute dimensionless error metrics
    eps = 1e-8
    pos_scale = train_stats['edge_feat_max'].item() + eps
    vel_scale = train_stats['node_vel_max'].item() + eps
    angvel_scale = train_stats['node_angvel_max'].item() + eps

    lst_error_pos, lst_error_vel, lst_error_angvel = [], [], []

    # In-memory trajectory trackers. Used for generating global physics panels (energy/momentum)
    predicted_traj_sys, grndtruth_traj_sys = [], []

    plot_folder_path = os.path.join(save_folder, experiment_name, 'rollout_plots')

    # Store data universally in a tmp folder first.
    data_folder_path = os.path.join(save_folder, experiment_name, 'tmp_rollout_data')
    if os.path.exists(data_folder_path): shutil.rmtree(data_folder_path)
    os.makedirs(data_folder_path, exist_ok=True)

    if plot:
        if os.path.exists(plot_folder_path): shutil.rmtree(plot_folder_path)
        os.makedirs(plot_folder_path, exist_ok=True)

    with torch.no_grad():
        # A persistent iterator is used instead of a standard `for` loop.
        # This allows manual advancement of the ground truth frames to precisely
        # synchronize with the internal micro-stepping clocks.
        loader_iter = iter(test_loader)

        # Fast-forward the loader to the requested start index
        for _ in range(start):
            try: next(loader_iter)
            except StopIteration: return [], [], [], [], []

        try:
            g0 = next(loader_iter).to(device)
            g_sync = g0
        except StopIteration:
            return [], [], [], [], []

        # Boundary models order real particles before ghosts, so the count of zero-valued
        # node features gives the split point and real particles can be sliced directly.
        N_real = (g0.node_feat.squeeze(1) == 0).sum().item()

        # Initialize the rollout state using the real particles from the first frame
        pos_real = g0.pos[:N_real]
        vel_real_t = g0.vel[:N_real]
        vel_real_tm1 = g0.prev_vel[:N_real]
        ang_real_t = g0.ang_vel[:N_real]
        ang_real_tm1 = g0.prev_ang_vel[:N_real]

        # Initialize orientation tracking for visualizations
        pred_angpos_sphere = torch.zeros_like(pos_real)
        pred_angpos_sphere[:, -1:] += 1
        gt_angpos_sphere = pred_angpos_sphere.clone()

        # Construct the initial full system graph (Real Particles + Boundary Ghosts)
        ingraph = interaction.create_graph(
            pos=pos_real, vel_t=vel_real_t, vel_tm1=vel_real_tm1,
            ang_vel_t=ang_real_t, ang_vel_tm1=ang_real_tm1,
            pos_tp1=pos_real, vel_tp1=vel_real_t, ang_vel_tp1=ang_real_t
        ).to(device)

        # ==========================================
        # SETUP SYNCHRONIZED TIME CLOCKS
        # ==========================================
        t_model = (start + 1) * dt_loader
        t_loader = (start + 1) * dt_loader

        done = False
        sync_idx = 0
        total_steps = end - start

        # 1. Autoregressive rollout.
        while not done and sync_idx < total_steps:

            # Calculate actual model steps
            current_model_step = (sync_idx * microsteps_per_sync) + 1
            total_model_steps = total_steps * microsteps_per_sync

            print(f"Evaluating model rollout step {current_model_step} / {total_model_steps} (Sim Time: {t_model:.5f}s)...")

            # 1. MICRO-STEPPING INTEGRATION LOOP (dt_model = 1e-4)
            for _ in range(microsteps_per_sync):
                if hasattr(interaction, 'rotating_walls'):
                    ingraph = interaction.rotating_walls(ingraph, sim_time=t_model, sim_time_previous=t_model - dt_model).to(device)

                dv_pred, dw_pred, dx_pred = model(ingraph)

                # Integrate kinematics strictly for the real particles using fast slicing
                ingraph.prev_vel[:N_real]     = ingraph.vel[:N_real]
                ingraph.vel[:N_real]          += dv_pred[:N_real]
                ingraph.prev_ang_vel[:N_real] = ingraph.ang_vel[:N_real]
                ingraph.ang_vel[:N_real]      += dw_pred[:N_real]
                ingraph.pos[:N_real]          += dx_pred[:N_real]

                # Update visual orientations based on internal model steps
                pred_angpos_sphere += 0.5 * (ingraph.ang_vel[:N_real] + ingraph.prev_ang_vel[:N_real]) * dt_model

                # Re-evaluate boundaries based on new positions to generate accurate graph topology
                ingraph = interaction.create_graph(
                    pos=ingraph.pos[:N_real],
                    vel_t=ingraph.vel[:N_real], vel_tm1=ingraph.prev_vel[:N_real],
                    ang_vel_t=ingraph.ang_vel[:N_real], ang_vel_tm1=ingraph.prev_ang_vel[:N_real],
                    pos_tp1=ingraph.pos[:N_real], vel_tp1=ingraph.vel[:N_real], ang_vel_tp1=ingraph.ang_vel[:N_real]
                ).to(device)

                t_model += dt_model

            # 2. TEMPORAL SYNCHRONIZATION (dt_loader = 1e-3)
            # Advance the dataset loader until its physical time matches the model's integrated time
            while abs(t_model - t_loader) > 1e-6:
                t_loader += dt_loader
                try:
                    g_sync = next(loader_iter)
                    g_sync = g_sync.to(device)

                    gt_angvel_sync = g_sync.ang_vel[:N_real]
                    gt_angpos_sphere += 0.5 * (gt_angvel_sync + g_sync.prev_ang_vel[:N_real]) * dt_loader
                except StopIteration:
                    done = True
                    break

            if done: break

            # Extract ground truth states for the now perfectly aligned timeframe
            gt_pos = g_sync.pos[:N_real]
            gt_vel = g_sync.vel[:N_real]
            gt_angvel = g_sync.ang_vel[:N_real]

            pred_pos_sphere = ingraph.pos[:N_real]
            pred_vel_sphere = ingraph.vel[:N_real]
            pred_angvel_sphere = ingraph.ang_vel[:N_real]

            # 3. METRICS
            error_pos = torch.mean(torch.abs((pred_pos_sphere - gt_pos) / pos_scale))
            error_vel = torch.mean(torch.abs((pred_vel_sphere - gt_vel) / vel_scale))
            error_angvel = torch.mean(torch.abs((pred_angvel_sphere - gt_angvel) / angvel_scale))

            abs_mae_pos = (pred_pos_sphere * 1000.0 - gt_pos * 1000.0).abs().mean()
            abs_mae_vel = (pred_vel_sphere - gt_vel).abs().mean()
            abs_mae_ang = (pred_angvel_sphere - gt_angvel).abs().mean()

            lst_error_pos.append(error_pos.item())
            lst_error_vel.append(error_vel.item())
            lst_error_angvel.append(error_angvel.item())

            # 4. UNIVERSAL CACHING
            # Always save data to the temp folder so the plotter has access to it later
            snapshot_data = {
                't_model': t_model,
                'pred_pos': pred_pos_sphere.cpu(),
                'pred_vel': pred_vel_sphere.cpu(),
                'pred_ang': pred_angvel_sphere.cpu(),
                'pred_angpos': pred_angpos_sphere.cpu(),
                'gt_pos': gt_pos.cpu(),
                'gt_vel': gt_vel.cpu(),
                'gt_ang': gt_angvel.cpu(),
                'gt_angpos': gt_angpos_sphere.cpu(),
                'mae_pos': abs_mae_pos.cpu(),
                'mae_vel': abs_mae_vel.cpu(),
                'mae_ang': abs_mae_ang.cpu()
            }
            torch.save(snapshot_data, os.path.join(data_folder_path, f"snapshot_{sync_idx:04d}_{t_model:.6f}.pt"))

            if not save_data:
                # Maintain full arrays in RAM if serialization is ultimately disabled
                predicted_traj_sys.append(torch.hstack((pred_pos_sphere, pred_vel_sphere, pred_angvel_sphere, pred_angpos_sphere)))
                grndtruth_traj_sys.append(torch.hstack((gt_pos, gt_vel, gt_angvel, gt_angpos_sphere)))

            sync_idx += 1

        # 2. Render frames from the snapshots, one process per core.
        if plot:
            print(f"\nPhysics rollout complete. Rendering frames to {plot_folder_path} using multiprocessing...")
            pt_files = sorted([os.path.join(data_folder_path, f) for f in os.listdir(data_folder_path) if f.endswith('.pt')])

            is_cylinder = hasattr(interaction, 'rotating_walls')
            if is_cylinder:
                # Convert to numpy explicitly to avoid sending PyTorch CUDA tensors across process boundaries
                bounds_dict = {
                    'center': interaction.center.cpu().numpy(),
                    'radius': interaction.radius.item(),
                    'length': interaction.length.item(),
                    'axis': interaction.axis.cpu().numpy()
                }
            else:
                bounds_dict = None

            # Prepare the tasks list (filters by frequency to avoid spinning up unnecessary processes)
            tasks = []
            for file_path in pt_files:
                f_idx = int(os.path.basename(file_path).split('_')[1])
                if f_idx % frequency == 0 or f_idx == 0:
                    tasks.append((file_path, plot_folder_path, f_idx, total_steps, is_cylinder, bounds_dict, plot_region, bottom_wall, experiment_name))

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
                    snap['pred_pos'], snap['pred_vel'], snap['pred_ang'], snap['pred_angpos']
                )))
                grndtruth_traj_sys.append(torch.hstack((
                    snap['gt_pos'], snap['gt_vel'], snap['gt_ang'], snap['gt_angpos']
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
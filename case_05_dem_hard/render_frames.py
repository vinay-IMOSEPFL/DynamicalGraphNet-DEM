"""
Standalone Multiprocessed Renderer for DEM Rollouts.

This script reads pre-computed `.pt` rollout snapshots from disk and uses
multiprocessing to rapidly generate 3D frame plots, compile an animated GIF,
and generate global physics conservation panels (Energy/Momentum).
"""

# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import argparse
import torch
import concurrent.futures
# Import visualizations and configurations
from visualization import plot_snapshot_cyl, plot_snapshot_test, create_gif
from config import GEO_DATA_CYLINDER, GEO_DATA_CUBOID
from boundary_model import insert_boundary

def _render_frame_worker(args):
    """
    Top-level worker function to render a single frame in a background CPU process.
    """
    file_path, plot_folder_path, f_idx, total_steps, is_cylinder, bounds_dict, plot_region, bottom_wall, experiment_name = args

    data = torch.load(file_path, map_location='cpu')

    # Safely handle key differences between simple and hard case data dicts
    pred_ang = data.get('pred_ang', data.get('pred_angvel'))
    gt_ang = data.get('gt_ang', data.get('gt_angvel'))
    mae_ang = data.get('mae_ang', data.get('mae_angvel'))

    if is_cylinder:
        plot_snapshot_cyl(
            predicted_node_pos=data['pred_pos'].numpy(),
            groundtruth_node_pos=data['gt_pos'].numpy(),
            sim_time=data.get('t_model', 0.0),
            frame_idx=f_idx,
            error_pos=data['mae_pos'].item(),
            error_vel=data['mae_vel'].item(),
            error_angvel=mae_ang.item(),
            plot_folder_path=plot_folder_path,
            boundaries=bounds_dict,
            model_name='Dynami-CAL'
        )
    else:
        plot_snapshot_test(
            predicted_node_pos=data['pred_pos'].numpy(),
            predicted_node_angpos=data['pred_angpos'].numpy(),
            groundtruth_node_pos=data['gt_pos'].numpy(),
            groundtruth_node_angpos=data['gt_angpos'].numpy(),
            predicted_node_vel=data['pred_vel'].numpy(),
            predicted_node_angvel=pred_ang.numpy(),
            groundtruth_node_vel=data['gt_vel'].numpy(),
            groundtruth_node_angvel=gt_ang.numpy(),
            error_pos=data['mae_pos'].item(),
            error_vel=data['mae_vel'].item(),
            error_angvel=mae_ang.item(),
            timestep=f_idx,
            plot_folder_path=plot_folder_path,
            experiment_name=experiment_name,
            plot_region=plot_region,
            bottom_wall=bottom_wall
        )

def render_rollout_multiprocessed(pt_files, plot_folder_path, total_files, is_cylinder, bounds_dict, plot_region, bottom_wall, experiment_name, frequency=10):
    """
    Packages the rendering tasks and dispatches them to the CPU pool.
    This is the function you import into your main_dem_hard.py script.
    """
    tasks = []
    for file_path in pt_files:
        f_idx = int(os.path.basename(file_path).split('_')[1].split('.')[0])
        if f_idx % frequency == 0 or f_idx == 0:
            tasks.append((file_path, plot_folder_path, f_idx, total_files, is_cylinder, bounds_dict, plot_region, bottom_wall, experiment_name))

    print(f"\nDispatching {len(tasks)} frame rendering tasks across {os.cpu_count()} CPU cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Use map to run them; printing is handled inside the worker
        for _ in executor.map(_render_frame_worker, tasks):
            pass


def main():
    """
    Command-line execution entry point.
    Allows you to render a folder of .pt files manually.
    """
    parser = argparse.ArgumentParser(description="Standalone Renderer for DEM Rollouts")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the folder containing .pt snapshot files")
    parser.add_argument('--cylinder', action='store_true', help="Flag to render the rotating cylinder case")
    parser.add_argument('--frequency', type=int, default=1, help="Render every Nth frame to speed up GIF creation")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    # Define output paths next to the data directory
    base_dir = os.path.dirname(args.data_dir)
    experiment_name = os.path.basename(base_dir)
    plot_dir = os.path.join(base_dir, 'rollout_plots')
    gif_dir = os.path.join(base_dir, 'gif')
    physics_dir = os.path.join(base_dir, 'physics_plots')

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(physics_dir, exist_ok=True)

    pt_files = sorted([os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pt')])
    total_files = len(pt_files)

    if total_files == 0:
        print("No .pt files found in the provided directory.")
        return

    print(f"Found {total_files} snapshots. Preparing multiprocessed rendering...")

    # Setup boundary data
    bounds_dict = None
    plot_region = (-0.03, 0.03)

    if args.cylinder:
        boundaries = insert_boundary(GEO_DATA_CYLINDER, boundary='cylinder', device='cpu')
        bounds_dict = {
            'center': boundaries['center'].numpy(),
            'radius': boundaries['radius'].item(),
            'length': boundaries['length'].item(),
            'axis': boundaries['axis'].numpy()
        }
    else:
        plot_region = (GEO_DATA_CUBOID[0][0], GEO_DATA_CUBOID[1][0])

    # Extract trajectories for physics panel generation
    predicted_traj_sys = []
    grndtruth_traj_sys = []

    print("\nExtracting trajectories for physics panel generation...")
    for file_path in pt_files:
        data = torch.load(file_path, map_location='cpu')
        p_ang = data.get('pred_ang', data.get('pred_angvel'))
        g_ang = data.get('gt_ang', data.get('gt_angvel'))

        predicted_traj_sys.append(torch.hstack((data['pred_pos'], data['pred_vel'], p_ang, data['pred_angpos'])))
        grndtruth_traj_sys.append(torch.hstack((data['gt_pos'], data['gt_vel'], g_ang, data['gt_angpos'])))

    # 1. Asynchronous Multiprocessed Plotting (calling the modular function)
    render_rollout_multiprocessed(
        pt_files=pt_files,
        plot_folder_path=plot_dir,
        total_files=total_files,
        is_cylinder=args.cylinder,
        bounds_dict=bounds_dict,
        plot_region=plot_region,
        bottom_wall=False,
        experiment_name=experiment_name,
        frequency=args.frequency
    )


    # 3. GIF Compilation
    print("\nCompiling GIF...")
    gif_name = f'{experiment_name}.gif'
    create_gif(folder=plot_dir, save_folder=gif_dir, gif_name=gif_name)
    print(f"Done! All visualizations saved to {base_dir}")

if __name__ == "__main__":
    main()
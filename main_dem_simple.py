"""
Training and evaluation entry point for the homogeneous DEM cases.

Covers 60-sphere systems inside a cuboidal enclosure with no external forces, plus two
isolated collision benchmarks used to probe the learned contact response directly.

Modes:
    train                         Train on the cuboid cases (case_01-case_05).
    test                          Autoregressive rollout on the held-out case_07.
    benchmark_sphere_collisions   Two-sphere oblique impact in free space.
    benchmark_wall_collisions     Single-sphere wall impact at 10, 30, 45, 60 and 90 degrees.
"""

import os
import shutil
import torch
import numpy as np
import random
import argparse
from torch_geometric.loader import DataLoader

# Core configuration and data structures
from case_04_dem_simple.config import DATASET_DIR, SAMPLE_TIME_STEP, THRESHOLD, MODEL_SETTINGS, GEO_DATA, SAVED_MODELS_DIR, RESULTS_DIR
from case_04_dem_simple.dataset import DemDataset
from case_04_dem_simple.boundary_model import SphereWallInteraction, insert_boundary

# Neural Network modules and utilities
from model.model_dem import DynamicsSolver
from utils.trainer_dem import Trainer, train_one_epoch
from case_04_dem_simple.rollout_evaluator import evaluate_rollout

def set_seed(seed=100):
    """
    Enforces reproducibility across runs by locking all random number generators.
    
    Args:
        seed (int): The fixed seed value to use for Python, NumPy, and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    """Parse arguments and dispatch to the selected mode."""
    parser = argparse.ArgumentParser(description="DEM Dynamics Solver Pipeline")
    parser.add_argument('--mode', type=str, default='train', 
                            choices=['train', 'test', 'benchmark_sphere_collisions', 'benchmark_wall_collisions'], 
                            help="Select the operational mode to execute.")
    parser.add_argument('--target_batch', type=int, default=MODEL_SETTINGS.get("batch_size", 64), 
                        help="Target effective batch size (used for gradient accumulation).")
    
    # Execution modifiers for visualization and data saving
    parser.add_argument('--plot', action='store_true', help="Enable 3D visualization plotting, GIF creation, and physics panels.")
    parser.add_argument('--save_plot', action='store_true', help="Keep individual PNG image frames after generating the compiled GIF.")
    parser.add_argument('--save_data', action='store_true', help="Save rollout trajectory data dictionary to a .pt file.")
    args = parser.parse_args()

    set_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Gradient accumulation logic allows training on smaller GPUs while simulating large batches
    actual_batch_size = MODEL_SETTINGS.get("batch_size", 64)
    accumulation_steps = max(1, args.target_batch // actual_batch_size)
    
    print(f"Device: {device} | Mode: {args.mode.upper()}")
    print(f"Loader Batch Size: {actual_batch_size} | Target Batch: {args.target_batch} | Accumulation Steps: {accumulation_steps}")

    # ==========================================================
    # 1. Dataset Initialization
    # ==========================================================
    print("Loading datasets...")
    train_dataset = DemDataset(DATASET_DIR, split="homogeneous/training")
    val_dataset = DemDataset(DATASET_DIR, split="homogeneous/validation")

    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Compute global normalization factors directly from the training set
    print("Computing training statistics...")
    train_stats = train_dataset.get_stats(device=device)

    # ==========================================================
    # 2. Model & Physics Initialization
    # ==========================================================
    model = DynamicsSolver(
        sample_step=SAMPLE_TIME_STEP, 
        train_stats=train_stats,
        num_msgs=MODEL_SETTINGS.get("num_msgs", 5),
        latent_size=MODEL_SETTINGS.get("nf", 128)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_SETTINGS.get("lr", 3e-4))

    # Setup the default physical boundaries for standard cuboidal cases
    boundaries = insert_boundary(GEO_DATA, device='cpu')
    interaction = SphereWallInteraction(boundaries, THRESHOLD, device='cpu')

    trainer = Trainer(
        model=model, 
        optimizer=optimizer, 
        device=device, 
        boundaries=boundaries, 
        threshold=THRESHOLD,
        train_stats=train_stats,
        time_step=SAMPLE_TIME_STEP
    )
    trainer.model_dir = SAVED_MODELS_DIR

    # ==========================================================
    # MODE: TRAIN
    # ==========================================================
    if args.mode == 'train':
        epochs = MODEL_SETTINGS.get("epochs", 600)
        best_val_loss = float('inf')
        eval_frequency = 5 

        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            train_loss = train_one_epoch(trainer, train_loader, pbar_desc=f"Epoch {epoch+1}/{epochs}", accumulation_steps=accumulation_steps)
            
            # Periodically evaluate the model's rollout stability on the validation set
            if (epoch + 1) % eval_frequency == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1} Train Loss (MSE): {train_loss:.4e}")
                
                val_pos_err, val_vel_err, val_angvel_err, _, _ = evaluate_rollout(
                    test_loader=val_loader,
                    model=model,
                    interaction=interaction,
                    device=device,
                    train_stats=train_stats,
                    time_step=SAMPLE_TIME_STEP,
                    start=0, end=200,
                    plot=False, save_data=False,
                    experiment_name='validation',
                    save_folder=RESULTS_DIR
                )
                
                # Combine physical errors to determine overall model health
                mean_val_pos = np.mean(val_pos_err)
                mean_val_vel = np.mean(val_vel_err)
                mean_val_angvel = np.mean(val_angvel_err)
                total_val_score = mean_val_vel + mean_val_angvel + mean_val_pos 
                
                print(f"Validation Score: {total_val_score:.4e} "
                      f"(Pos: {mean_val_pos:.4e}, Vel: {mean_val_vel:.4e}, AngVel: {mean_val_angvel:.4e})")
                
                # Checkpointing
                if total_val_score < best_val_loss:
                    best_val_loss = total_val_score
                    trainer.save_model("best_val")
                    print(f"--> New best model saved to {trainer.model_dir}! (Score: {best_val_loss:.4e})")
                    
    # ==========================================================
    # MODE: TEST (Standard Extrapolation)
    # ==========================================================
    elif args.mode == 'test':
        case_name = "case_07"
        
        expt_dataset = DemDataset(DATASET_DIR, case_name=case_name)
        expt_loader = DataLoader(expt_dataset, batch_size=1, shuffle=False)
        
        # Restore the best weights from training
        best_model_path = os.path.join(trainer.model_dir, "model_checkpoint_best_val.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best validation model from {best_model_path}")
        else:
            print(f"Warning: Checkpoint not found at {best_model_path}. Evaluating with uninitialized weights.")
        
        print(f"\nStarting Final Rollout on {case_name} (Plotting: {args.plot} | Save Data: {args.save_data})...")
        
        plot_limit_upper = GEO_DATA[1][0] if len(GEO_DATA) > 1 else 0.03

        pos_err, vel_err, angvel_err, predicted_traj_sys, grndtruth_traj_sys = evaluate_rollout(
            test_loader=expt_loader,
            model=model,
            interaction=interaction,
            device=device,
            train_stats=train_stats,
            time_step=SAMPLE_TIME_STEP,
            start=0, end=len(expt_loader),  
            plot=args.plot,    
            save_data=args.save_data,
            frequency=50,      
            experiment_name=f'{case_name}_rollout',
            save_folder=RESULTS_DIR,
            plot_region=(GEO_DATA[0][0], plot_limit_upper),
            bottom_wall=False
        )
        
        # Post-process visualization artifacts
        if args.plot:
            from case_04_dem_simple.visualization import create_gif, plot_physics_panel
            
            print("\nGenerating GIF...")
            plot_folder_path = os.path.join(RESULTS_DIR, f'{case_name}_rollout', 'rollout_plots')
            gif_save_dir = os.path.join(RESULTS_DIR, f'{case_name}_rollout', 'gif')
            
            os.makedirs(gif_save_dir, exist_ok=True)
            create_gif(
                folder=plot_folder_path, 
                save_folder=gif_save_dir, 
                gif_name='trajectory_rollout_DynSolver.gif'
            )
            print(f"Saved GIF to {gif_save_dir}")
            
            print("Generating Production Physics Panel...")
            physics_plot_dir = os.path.join(RESULTS_DIR, f'{case_name}_rollout', 'rollout_physics_plots')
            plot_physics_panel(
                predicted_traj_sys, 
                grndtruth_traj_sys, 
                system_name="Test_Rollout", 
                save_dir=physics_plot_dir
            )
            print(f"Saved physics panel to {physics_plot_dir}")

            # Clean up intermediate image files to save disk space
            if not args.save_plot:
                if os.path.exists(plot_folder_path):
                    shutil.rmtree(plot_folder_path)
                    print(f"Deleted temp PNG frames in {plot_folder_path} as --save_plot was False.")
        
        print("\nTest execution complete.")

    # ==========================================================
    # BENCHMARK: OBLIQUE SPHERE COLLISIONS
    # ==========================================================
    elif args.mode == 'benchmark_sphere_collisions':
        case_name = "oblique_sphere_collisions"
        expt_dataset = DemDataset(DATASET_DIR, case_name=case_name)
        expt_loader = DataLoader(expt_dataset, batch_size=1, shuffle=False)
        
        best_model_path = os.path.join(trainer.model_dir, "model_checkpoint_best_val.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from {best_model_path}")
        
        # Force a massively expanded bounding box so the two spheres only interact with each other,
        # effectively simulating free space dynamics.
        geo_sphere = ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0))
        bounds_sphere = insert_boundary(geo_sphere, device='cpu')
        interaction_sphere = SphereWallInteraction(bounds_sphere, THRESHOLD, device='cpu')
        
        print(f"\nStarting Rollout: Oblique Sphere Collisions...")
        _, _, _, pred_traj, gt_traj = evaluate_rollout(
            test_loader=expt_loader, model=model, interaction=interaction_sphere,
            device=device, train_stats=train_stats, time_step=SAMPLE_TIME_STEP,
            start=0, end=len(expt_loader), plot=args.plot, save_data=args.save_data, frequency=5,
            experiment_name=f'benchmark_{case_name}', save_folder=RESULTS_DIR
        )

        if args.plot:
            from case_04_dem_simple.visualization import create_gif, plot_physics_panel
            print("Generating Physics Panel & GIF...")
            plot_dir = os.path.join(RESULTS_DIR, f'benchmark_{case_name}')
            
            os.makedirs(os.path.join(plot_dir, 'gif'), exist_ok=True)
            create_gif(folder=os.path.join(plot_dir, 'rollout_plots'), 
                       save_folder=os.path.join(plot_dir, 'gif'), gif_name='oblique_sphere.gif')
            
            plot_physics_panel(pred_traj, gt_traj, "Oblique Sphere", os.path.join(plot_dir, 'physics_plots'))
            
            if not args.save_plot:
                shutil.rmtree(os.path.join(plot_dir, 'rollout_plots'), ignore_errors=True)

    # ==========================================================
    # BENCHMARK: OBLIQUE WALL COLLISIONS (10, 30, 45, 60, 90 deg)
    # ==========================================================
    elif args.mode == 'benchmark_wall_collisions':
        angles = [10, 30, 45, 60, 90]
        
        best_model_path = os.path.join(trainer.model_dir, "model_checkpoint_best_val.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from {best_model_path}")
        
        # Override the boundaries to feature a single solid plane at Z=0.
        geo_oblique = ((-100.0, -100.0, 0.0), (100.0, 100.0, 100.0))
        bounds_oblique = insert_boundary(geo_oblique, device='cpu')
        threshold_oblique = 1.25 * 0.005 
        interaction_oblique = SphereWallInteraction(bounds_oblique, threshold_oblique, device='cpu')

        pred_ang_vel_list = []
        gt_ang_vel_list = []
        
        print(f"\nStarting Oblique Wall Impact Evaluation Across {len(angles)} Angles...")
        
        for angle in angles:
            case_name = os.path.join("oblique_wall_collisions", f"{angle}_deg")
            print(f"Evaluating Impact Angle: {angle}°")
            
            expt_dataset = DemDataset(DATASET_DIR, case_name=case_name)
            expt_loader = DataLoader(expt_dataset, batch_size=1, shuffle=False)
            
            _, _, _, pred_traj, gt_traj = evaluate_rollout(
                test_loader=expt_loader, model=model, interaction=interaction_oblique,
                device=device, train_stats=train_stats, time_step=SAMPLE_TIME_STEP,
                start=0, end=len(expt_loader), plot=False, save_data=False, 
                experiment_name=f'benchmark_oblique_{angle}deg', save_folder=RESULTS_DIR
            )
            
            # Extract the post-collision angular velocity from the final predicted timestep
            # Tensor Shape: [num_spheres, 12], where columns 6:9 represent angular velocity (wx, wy, wz)
            pred_w = pred_traj[-1][0, 6:9].cpu().numpy()
            gt_w = gt_traj[-1][0, 6:9].cpu().numpy()
            
            pred_ang_vel_list.append(np.hstack(([angle], pred_w)))
            gt_ang_vel_list.append(np.hstack(([angle], gt_w)))
            
        # Compile all angles into a single array for visualization
        pred_arr = np.vstack(pred_ang_vel_list)
        gt_arr = np.vstack(gt_ang_vel_list)
        
        from case_04_dem_simple.visualization import plot_angular_velocity_components
        
        save_dir = os.path.join(RESULTS_DIR, 'benchmark_oblique_summary')
        print(f"\nGenerating Oblique Impact Summary Plot at {save_dir}")
        plot_angular_velocity_components(pred_arr, gt_arr, save_dir)
        print("Done!")

if __name__ == "__main__":
    main()
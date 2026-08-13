"""
Training and evaluation entry point for the heterogeneous DEM cases.

Covers systems under gravity, which require the external-force MLP, and adds two
capabilities absent from the homogeneous pipeline: rotating cylindrical boundaries,
and micro-stepping to bridge the model timestep (1e-4 s) and the coarser interval at
which the cylinder ground truth was saved (1e-3 s).

Modes:
    train       Train on the gravity cuboid cases (case_01-case_05).
    test        Autoregressive rollout on the cuboid case_06.
    cylinder    Extrapolation rollout in the rotating cylinder (2,073 spheres).
"""

import os
import shutil
import torch
import numpy as np
import random
import argparse
from torch_geometric.loader import DataLoader

# Core configuration and data structures
from case_05_dem_hard.config import (
    DATASET_DIR, MODEL_SETTINGS, THRESHOLD, SAVED_MODELS_DIR, RESULTS_DIR,
    GEO_DATA_CUBOID, SAMPLE_TIME_STEP_CUBOID,
    GEO_DATA_CYLINDER, SAMPLE_TIME_STEP_CYLINDER
)
from case_05_dem_hard.dataset import DemDataset
from case_05_dem_hard.boundary_model import SphereWallInteraction, CylinderInteraction, insert_boundary
from case_05_dem_hard.rollout_evaluator import evaluate_rollout

# Neural Network modules and utilities
from model.model_dem import DynamicsSolver
from utils.trainer_dem import Trainer, train_one_epoch


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
    parser = argparse.ArgumentParser(description="DEM Dynamics Solver Pipeline - Gravity & Cylinder")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'cylinder'],
                        help="Select mode to execute (train cuboid, test cuboid, or extrapolate cylinder).")
    parser.add_argument('--target_batch', type=int, default=MODEL_SETTINGS.get("batch_size", 64),
                        help="Target effective batch size (used for gradient accumulation).")

    # Execution modifiers for visualization and data saving
    parser.add_argument('--plot', action='store_true', help="Enable visualization plotting, GIF creation, and physics panels.")
    parser.add_argument('--save_plot', action='store_true', help="Keep PNG image frames after generating the GIF.")
    parser.add_argument('--save_data', action='store_true', help="Save rollout trajectory data dictionary to .pt file.")
    args = parser.parse_args()

    set_seed(100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gradient accumulation allows training on smaller GPUs while simulating large batches
    actual_batch_size = MODEL_SETTINGS.get("batch_size", 64)
    accumulation_steps = max(1, args.target_batch // actual_batch_size)

    print(f"Device: {device} | Mode: {args.mode.upper()}")
    print(f"Loader Batch Size: {actual_batch_size} | Target Batch: {args.target_batch} | Accumulation Steps: {accumulation_steps}")

    # ==========================================================
    # 1. Dataset Initialization
    # ==========================================================
    print("Loading base gravity datasets...")
    train_dataset = DemDataset(DATASET_DIR, split="heterogeneous/gravity/training")
    val_dataset   = DemDataset(DATASET_DIR, split="heterogeneous/gravity/validation")

    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Compute global normalization factors directly from the training set
    print("Computing training statistics...")
    train_stats = train_dataset.get_stats(device=device)

    # ==========================================================
    # 2. Model & Physics Initialization
    # ==========================================================
    # Initialize the core Graph Neural Network.
    # Passing ext_force=True enables the external gravity MLP within the network.
    model = DynamicsSolver(
        sample_step=SAMPLE_TIME_STEP_CUBOID,
        train_stats=train_stats,
        num_msgs=MODEL_SETTINGS.get("num_msgs", 5),
        latent_size=MODEL_SETTINGS.get("nf", 128),
        ext_force=MODEL_SETTINGS.get("use_ext_force", True)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=MODEL_SETTINGS.get("lr", 3e-4))

    # Setup the default physical boundaries for standard cuboidal gravity cases
    boundaries = insert_boundary(GEO_DATA_CUBOID, boundary='cuboid', device=device)
    interaction = SphereWallInteraction(boundaries, THRESHOLD, device=device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        boundaries=boundaries,
        threshold=THRESHOLD,
        train_stats=train_stats,
        time_step=SAMPLE_TIME_STEP_CUBOID
    )
    trainer.model_dir = SAVED_MODELS_DIR

    # ==========================================================
    # MODE: TRAIN (Gravity Cuboid)
    # ==========================================================
    if args.mode == 'train':
        epochs = MODEL_SETTINGS.get("epochs", 600)
        best_val_loss = float('inf')
        eval_frequency = 5

        print(f"Training for {epochs} epochs with Gravity Network Enabled...")
        for epoch in range(epochs):
            train_loss = train_one_epoch(trainer, train_loader, pbar_desc=f"Epoch {epoch+1}/{epochs}", accumulation_steps=accumulation_steps)

            # Periodically evaluate the model's rollout stability on the validation set
            if (epoch + 1) % eval_frequency == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1} Train Loss (MSE): {train_loss:.4e}")

                # Standard 1:1 timestep syncing is used for cuboid evaluations
                val_pos_err, val_vel_err, val_angvel_err, _, _ = evaluate_rollout(
                    test_loader=val_loader,
                    model=model,
                    interaction=interaction,
                    device=device,
                    train_stats=train_stats,
                    dt_model=SAMPLE_TIME_STEP_CUBOID,
                    dt_loader=SAMPLE_TIME_STEP_CUBOID,
                    microsteps_per_sync=1,
                    start=0, end=200,
                    plot=False, save_data=False,
                    experiment_name='validation_gravity',
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
    # MODE: TEST (Gravity Cuboid Extrapolation)
    # ==========================================================
    elif args.mode == 'test':
        case_name = "case_06"

        expt_dataset = DemDataset(DATASET_DIR, case_name=case_name)
        expt_loader = DataLoader(expt_dataset, batch_size=1, shuffle=False)

        # Restore the best weights from training
        best_model_path = os.path.join(trainer.model_dir, "model_checkpoint_best_val.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best validation model from {best_model_path}")

        print(f"\nStarting Gravity Cuboid Rollout ({case_name})...")

        pos_err, vel_err, angvel_err, predicted_traj_sys, grndtruth_traj_sys = evaluate_rollout(
            test_loader=expt_loader,
            model=model,
            interaction=interaction,
            device=device,
            train_stats=train_stats,
            dt_model=SAMPLE_TIME_STEP_CUBOID,
            dt_loader=SAMPLE_TIME_STEP_CUBOID,
            microsteps_per_sync=1,
            start=0, end=len(expt_loader),
            plot=args.plot,
            save_data=args.save_data,
            frequency=50,
            experiment_name=f'{case_name}_gravity_rollout',
            save_folder=RESULTS_DIR,
            plot_region=(GEO_DATA_CUBOID[0][0], GEO_DATA_CUBOID[1][0]),
            bottom_wall=False
        )

        # Post-process visualization artifacts
        if args.plot:
            from case_05_dem_hard.visualization import create_gif, plot_physics_panel

            print("\nGenerating GIF...")
            plot_folder_path = os.path.join(RESULTS_DIR, f'{case_name}_gravity_rollout', 'rollout_plots')
            gif_save_dir = os.path.join(RESULTS_DIR, f'{case_name}_gravity_rollout', 'gif')
            os.makedirs(gif_save_dir, exist_ok=True)
            create_gif(folder=plot_folder_path, save_folder=gif_save_dir, gif_name=f'{case_name}_gravity.gif')

            print("Generating Physics Panel...")
            physics_plot_dir = os.path.join(RESULTS_DIR, f'{case_name}_gravity_rollout', 'physics_plots')
            plot_physics_panel(predicted_traj_sys, grndtruth_traj_sys, system_name=f"{case_name}_Gravity", save_dir=physics_plot_dir)

            # Clean up intermediate image files to save disk space
            if not args.save_plot:
                shutil.rmtree(plot_folder_path, ignore_errors=True)

    # ==========================================================
    # MODE: CYLINDER (Rotating Cylinder Extrapolation)
    # ==========================================================
    elif args.mode == 'cylinder':
        case_name = "rotating_cylinder"

        cyl_dataset = DemDataset(DATASET_DIR, case_name=case_name)
        cyl_loader = DataLoader(cyl_dataset, batch_size=1, shuffle=False)

        best_model_path = os.path.join(trainer.model_dir, "model_checkpoint_best_val.pth")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best validation model from {best_model_path}")

        # Swap the interaction model to handle cylindrical boundaries
        cyl_bounds = insert_boundary(GEO_DATA_CYLINDER, boundary='cylinder', device=device)
        cyl_interaction = CylinderInteraction(cyl_bounds, threshold=THRESHOLD, device=device)

        # Calculate integration synchronization steps
        # E.g., if model steps at 1e-4s but data is saved at 1e-3s, we need 10 microsteps per frame.
        microsteps = int(SAMPLE_TIME_STEP_CYLINDER / SAMPLE_TIME_STEP_CUBOID)

        print(f"\nStarting rotating cylinder rollout ({case_name})...")
        print(f"Syncing every {microsteps} micro-steps (Loader: {SAMPLE_TIME_STEP_CYLINDER}s | Model: {SAMPLE_TIME_STEP_CUBOID}s)")

        # Execute evaluating rollout, which natively supports dynamic rotating boundaries
        pos_err, vel_err, angvel_err, predicted_traj_sys, grndtruth_traj_sys = evaluate_rollout(
            test_loader=cyl_loader,
            model=model,
            interaction=cyl_interaction,
            device=device,
            train_stats=train_stats,
            dt_model=SAMPLE_TIME_STEP_CUBOID,
            dt_loader=SAMPLE_TIME_STEP_CYLINDER,
            microsteps_per_sync=microsteps,
            start=0, end=len(cyl_loader),
            plot=args.plot,
            save_data=args.save_data,
            frequency=25,
            experiment_name=f'{case_name}_rollout',
            save_folder=RESULTS_DIR
        )

        if args.plot:
            from case_05_dem_hard.visualization import create_gif

            print("\nGenerating Cylinder GIF...")
            plot_folder_path = os.path.join(RESULTS_DIR, f'{case_name}_rollout', 'rollout_plots')
            gif_save_dir = os.path.join(RESULTS_DIR, f'{case_name}_rollout', 'gif')
            os.makedirs(gif_save_dir, exist_ok=True)
            create_gif(folder=plot_folder_path, save_folder=gif_save_dir, gif_name=f'{case_name}.gif')


            if not args.save_plot:
                shutil.rmtree(plot_folder_path, ignore_errors=True)

if __name__ == "__main__":
    main()
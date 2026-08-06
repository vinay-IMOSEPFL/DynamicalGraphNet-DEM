"""
Data Preprocessing Pipeline for Heterogeneous DEM Datasets (Gravity & Cylinder).

This script processes raw Discrete Element Method (DEM) simulation data (CSVs) 
into PyTorch Geometric graph structures. It extracts kinematic properties, 
applies appropriate physical boundaries (cuboidal or cylindrical), and saves 
the processed sequences as serialized tensor files (.pt) for efficient training.
"""

import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# Append project root to path to allow absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from case_05_dem_hard.config import DATA_DIR, DATASET_DIR, THRESHOLD, GEO_DATA_CUBOID, GEO_DATA_CYLINDER, SPLIT_DIRS
from case_05_dem_hard.boundary_model import SphereWallInteraction, CylinderInteraction, insert_boundary

def read_time_step(data_path, time_step):
    """
    Reads raw DEM simulation data from a CSV file for a specific time step.
    
    Extracts the kinematic properties (position, velocity, angular velocity) 
    and handles material mapping based on particle density.

    Args:
        data_path (str): The directory containing the CSV files.
        time_step (int): The specific integer time step to read.

    Returns:
        tuple: PyTorch tensors for position, angular position, velocity, 
               angular velocity, and sphere type (material label). 
               All spatial/kinematic tensors have shape (N, 3).
    """
    filename = f'data_at_timestep_{time_step:03d}.csv'
    filepath = os.path.join(data_path, filename)
    df = pd.read_csv(filepath)
    
    # Extract 3D coordinates
    pos = torch.hstack((torch.tensor(df['coordinates:0'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['coordinates:1'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['coordinates:2'].astype(float)).reshape(-1, 1)))
    
    # Initialize angular position to zero (typically not exported by raw DEM output)
    ang_pos = torch.zeros_like(pos)
    
    # Extract 3D translational velocities
    vel = torch.hstack((torch.tensor(df['Velocity:0'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['Velocity:1'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['Velocity:2'].astype(float)).reshape(-1, 1)))
    
    # Extract 3D angular velocities
    ang_vel = torch.hstack((torch.tensor(df['Angular_velocity:0'].astype(float)).reshape(-1, 1), 
                            torch.tensor(df['Angular_velocity:1'].astype(float)).reshape(-1, 1), 
                            torch.tensor(df['Angular_velocity:2'].astype(float)).reshape(-1, 1)))
    
    # Map physical particle density to a categorical node type/material label
    if 'Density' in df.columns:
        unique_densities = df['Density'].astype(float).unique()
        density_label_map = {val: idx + 1 for idx, val in enumerate(sorted(unique_densities))}
        sphere_type_labels = df['Density'].astype(float).map(density_label_map).values
        sphere_type = torch.tensor(sphere_type_labels).reshape(-1, 1)
    else:
        # Fallback to a single homogeneous material if density is missing
        sphere_type = torch.zeros((df.shape[0], 1), dtype=torch.float32)

    return pos, ang_pos, vel, ang_vel, sphere_type

def extract_number(fn):
    """Parses the integer time step from a standard DEM filename (e.g., 'data_at_timestep_015.csv')."""
    try: 
        return int(fn.split('.')[0].split('_')[-1])
    except: 
        return None

def get_time_steps(data_path):
    """Scans a directory and returns a sorted list of available integer time steps."""
    fns = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    nums = [n for n in (extract_number(f) for f in fns) if n is not None]
    return sorted(nums)

def process_directory_tree(raw_root, save_root, interaction):
    """
    Recursively crawls a raw data directory, builds PyTorch Geometric graphs 
    representing the physical states using a sliding window, and saves them to disk.

    Args:
        raw_root (str): The root directory containing raw CSV data.
        save_root (str): The root directory where processed .pt files will be saved.
        interaction (object): The boundary interaction model used to construct the graph topology.
    """
    if not os.path.exists(raw_root): 
        print(f"  [Skipped] Source not found: {raw_root}")
        return
    
    for dirpath, dirnames, filenames in os.walk(raw_root):
        # Only process directories containing simulation frames
        if any(f.endswith('.csv') for f in filenames):
            rel_path = os.path.relpath(dirpath, raw_root)
            save_dir = os.path.normpath(os.path.join(save_root, rel_path))
            os.makedirs(save_dir, exist_ok=True)
            
            time_steps = get_time_steps(dirpath)
            if len(time_steps) < 3:
                print(f"Not enough timesteps in {dirpath}. Skipping.")
                continue
                
            # Exclude the first and last frames to allow for a (t-1, t, t+1) sliding window
            time_steps = time_steps[1:-1]
            all_graphs = []
            
            for t in tqdm(time_steps, desc=os.path.basename(dirpath), unit="step"):
                # Extract state at t (current), t-1 (history), and t+1 (target label)
                pos, _, vel_t, ang_vel_t, _ = read_time_step(dirpath, t)
                _, _, vel_tm1, ang_vel_tm1, _ = read_time_step(dirpath, t-1)
                pos_p1, _, vel_tp1, ang_vel_tp1, _ = read_time_step(dirpath, t+1)

                # Delegate graph topology construction and ghost node generation to the boundary model
                g = interaction.create_graph(
                    pos, vel_t, vel_tm1, ang_vel_t, ang_vel_tm1,
                    pos_p1, vel_tp1, ang_vel_tp1
                )
                all_graphs.append(g)

            # Serialize the sequence of processed graphs
            out_path = os.path.join(save_dir, "graph_list.pt")
            torch.save(all_graphs, out_path)
            print(f"Saved {len(all_graphs)} graphs -> {out_path}")

if __name__ == "__main__":
    print("==================================================")
    print(" PREPROCESSING HETEROGENEOUS GRAVITY DATASETS")
    print("==================================================")
    
    # =========================================================
    # 1. CUBOID CASES (Train, Val, Test)
    # =========================================================
    print("\n--- 1. Gravity Cuboid Datasets ---")
    
    # Initialize the boundary interaction model for the standard cuboidal enclosure
    cuboid_bounds = insert_boundary(GEO_DATA_CUBOID, boundary='cuboid', device='cpu')
    cuboid_interaction = SphereWallInteraction(cuboid_bounds, THRESHOLD, device='cpu')

    for split in ['train', 'val', 'test']:
        for rel_path in SPLIT_DIRS[split]:
            raw_dir = os.path.join(DATA_DIR, rel_path)
            save_dir = os.path.join(DATASET_DIR, rel_path)
            process_directory_tree(raw_dir, save_dir, cuboid_interaction)

    # =========================================================
    # 2. CYLINDER EXTRAPOLATION
    # =========================================================
    print("\n--- 2. Rotating Cylinder Extrapolation ---")
    
    # Switch boundary models to handle the curved, rotating cylinder geometry
    cyl_bounds = insert_boundary(GEO_DATA_CYLINDER, boundary='cylinder', device='cpu')
    cyl_interaction = CylinderInteraction(cyl_bounds, threshold=THRESHOLD, device='cpu')

    for rel_path in SPLIT_DIRS['cylinder']:
        raw_dir = os.path.join(DATA_DIR, rel_path)
        save_dir = os.path.join(DATASET_DIR, rel_path)
        process_directory_tree(raw_dir, save_dir, cyl_interaction)

    print("\nAll gravity preprocessing complete!")
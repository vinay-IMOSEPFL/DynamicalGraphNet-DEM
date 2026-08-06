import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

# Ensure we can import the module namespace if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from case_04_dem_simple.config import DATA_DIR, DATASET_DIR, THRESHOLD, GEO_DATA
from case_04_dem_simple.boundary_model import SphereWallInteraction, insert_boundary

def read_time_step(data_path, time_step):
    """
    Reads the raw DEM simulation data from a CSV file for a specific time step.
    
    Extracts the kinematic properties (position, velocity, angular velocity) 
    and (not used-->)handles material mapping if multiple sphere densities are present.

    Args:
        data_path (str): The directory containing the CSV files.
        time_step (int): The specific integer time step to read.

    Returns:
        tuple: Tensors for position, angular position (initialized to zero), 
               velocity, angular velocity, and sphere type (material label).
    """
    filename = f'data_at_timestep_{time_step:03d}.csv'
    filepath = os.path.join(data_path, filename)
    df = pd.read_csv(filepath)
    
    # Extract 3D coordinates into a shape (N, 3) tensor
    pos = torch.hstack((torch.tensor(df['coordinates:0'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['coordinates:1'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['coordinates:2'].astype(float)).reshape(-1, 1)))
    
    # Angular position is generally not provided by the raw DEM output, initialize to zero
    ang_pos = torch.zeros_like(pos)
    
    # Extract 3D translational velocities into a shape (N, 3) tensor
    vel = torch.hstack((torch.tensor(df['Velocity:0'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['Velocity:1'].astype(float)).reshape(-1, 1), 
                        torch.tensor(df['Velocity:2'].astype(float)).reshape(-1, 1)))
    
    # Extract 3D angular velocities into a shape (N, 3) tensor
    ang_vel = torch.hstack((torch.tensor(df['Angular_velocity:0'].astype(float)).reshape(-1, 1), 
                            torch.tensor(df['Angular_velocity:1'].astype(float)).reshape(-1, 1), 
                            torch.tensor(df['Angular_velocity:2'].astype(float)).reshape(-1, 1)))
    
    # Map particle density to a categorical node type (e.g., 1 for material A, 2 for material B)
    if 'Density' in df.columns:
        unique_densities = df['Density'].astype(float).unique()
        density_label_map = {val: idx + 1 for idx, val in enumerate(sorted(unique_densities))}
        sphere_type_labels = df['Density'].astype(float).map(density_label_map).values
        sphere_type = torch.tensor(sphere_type_labels).reshape(-1, 1)
    else:
        # Default all particles to type 0 if no density column is provided
        sphere_type = torch.zeros((df.shape[0], 1), dtype=torch.float32)

    return pos, ang_pos, vel, ang_vel, sphere_type

def extract_number(fn):
    """
    Parses the integer time step from a standard DEM filename.
    e.g., 'data_at_timestep_015.csv' -> 15
    """
    try: 
        return int(fn.split('.')[0].split('_')[-1])
    except: 
        return None

def get_time_steps(data_path):
    """
    Scans a directory for CSV files and returns a sorted list of available time steps.
    """
    fns = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    nums = [n for n in (extract_number(f) for f in fns) if n is not None]
    return sorted(nums)


if __name__ == "__main__":
    print("==================================================")
    print(" PREPROCESSING ALL DEM DATASETS")
    print("==================================================")
    
    def process_directory_tree(raw_root, save_root, interaction):
        """
        Recursively crawls a raw data directory, detects sequences of CSV files, 
        builds PyTorch Geometric graphs representing the physical states, 
        and saves them out as serialized PyTorch tensors.

        Args:
            raw_root (str): The root directory containing raw CSV data.
            save_root (str): The root directory where processed .pt files will be saved.
            interaction (object): The boundary interaction model used to construct the graphs.
        """
        if not os.path.exists(raw_root): 
            return
        
        for dirpath, dirnames, filenames in os.walk(raw_root):
            # Only process directories that actually contain CSV simulation frames
            if any(f.endswith('.csv') for f in filenames):
                rel_path = os.path.relpath(dirpath, raw_root)
                save_dir = os.path.normpath(os.path.join(save_root, rel_path))
                os.makedirs(save_dir, exist_ok=True)
                
                print(f"\nProcessing: {save_dir}")
                
                time_steps = get_time_steps(dirpath)
                if len(time_steps) < 3:
                    print(f"Not enough timesteps in {dirpath}. Skipping.")
                    continue
                    
                # We need t-1 (history) and t+1 (target), so we drop the first and last available frames
                time_steps = time_steps[1:-1]
                all_graphs = []
                
                # Sliding window approach to capture the full kinematic state and ground truth targets
                for t in tqdm(time_steps, desc=os.path.basename(dirpath), unit="step"):
                    pos, _, vel_t, ang_vel_t, _ = read_time_step(dirpath, t)
                    _, _, vel_tm1, ang_vel_tm1, _ = read_time_step(dirpath, t-1)
                    pos_p1, _, vel_tp1, ang_vel_tp1, _ = read_time_step(dirpath, t+1)

                    # Let the interaction model process the boundaries and build the graph topology
                    g = interaction.create_graph(
                        pos, vel_t, vel_tm1, ang_vel_t, ang_vel_tm1,
                        pos_p1, vel_tp1, ang_vel_tp1
                    )
                    all_graphs.append(g)

                # Save the complete sequence of processed graphs for this case
                out_path = os.path.join(save_dir, "graph_list.pt")
                torch.save(all_graphs, out_path)
                print(f"Saved {len(all_graphs)} graphs -> {out_path}")

    # Initialize the standard cuboidal boundaries for the main datasets
    main_boundaries = insert_boundary(GEO_DATA, device='cpu')
    main_interaction = SphereWallInteraction(main_boundaries, THRESHOLD, device='cpu')

    # ---------------------------------------------------------
    # 1. HOMOGENEOUS MAIN DATASETS
    # ---------------------------------------------------------
    print("\n--- 1. Homogeneous Datasets (Training / Validation / Extrapolation) ---")
    homo_raw = os.path.join(DATA_DIR, "homogeneous")
    homo_save = os.path.join(DATASET_DIR, "homogeneous")
    
    for split in ['training', 'validation', 'extrapolation']:
        split_raw = os.path.join(homo_raw, split)
        split_save = os.path.join(homo_save, split)
        if os.path.exists(split_raw):
            process_directory_tree(split_raw, split_save, main_interaction)

    # ---------------------------------------------------------
    # 3. BENCHMARK: OBLIQUE SPHERE COLLISIONS
    # ---------------------------------------------------------
    print("\n--- 3. Benchmark: Oblique Sphere Collisions ---")
    sphere_raw = os.path.join(homo_raw, "benchmark", "oblique_sphere_collisions")
    sphere_save = os.path.join(homo_save, "benchmark", "oblique_sphere_collisions")
    
    if os.path.exists(sphere_raw):
        # Use an effectively infinite bounding box so the spheres only interact with each other
        geo_sphere = ((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0))
        bounds_sphere = insert_boundary(geo_sphere, device='cpu')
        interaction_sphere = SphereWallInteraction(bounds_sphere, threshold=THRESHOLD, device='cpu')
        process_directory_tree(sphere_raw, sphere_save, interaction_sphere)
    else:
        print(f"Oblique Sphere collisions not found. Skipping...")

    # ---------------------------------------------------------
    # 4. BENCHMARK: OBLIQUE WALL COLLISIONS
    # ---------------------------------------------------------
    print("\n--- 4. Benchmark: Oblique Wall Collisions ---")
    wall_raw = os.path.join(homo_raw, "benchmark", "oblique_wall_collisions")
    wall_save = os.path.join(homo_save, "benchmark", "oblique_wall_collisions")
    
    if os.path.exists(wall_raw):
        # Configure a single boundary plane at Z=0 for the wall collision benchmark
        geo_wall = ((-100.0, -100.0, 0.0), (100.0, 100.0, 100.0))
        bounds_wall = insert_boundary(geo_wall, device='cpu')
        
        # Slightly scaled threshold to ensure accurate edge construction near the wall
        threshold_wall = 1.25 * 0.005 
        interaction_wall = SphereWallInteraction(bounds_wall, threshold_wall, device='cpu')
        process_directory_tree(wall_raw, wall_save, interaction_wall)
    else:
        print(f"Oblique Wall collisions not found. Skipping...")

    print("\nAll preprocessing complete!")
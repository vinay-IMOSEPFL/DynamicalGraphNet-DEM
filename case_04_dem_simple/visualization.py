import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import seaborn as sns
import imageio.v2 as imageio

# Milliseconds per GIF frame. 0.5 was interpreted as 0.5 ms and rounded to zero,
# which made every animation play at the viewer's maximum rate.
FRAME_DURATION_MS = 120


def set_plot_style():
    """Applies a consistent, publication-ready styling to all Seaborn/Matplotlib plots."""
    sns.set(
        context="paper",
        style="whitegrid",
        palette="deep",
        font="serif",
        font_scale=1.5,
        rc={"figure.figsize": (16, 10),
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "legend.title_fontsize": 16,
            "grid.linewidth": 0.8,
            "axes.linewidth": 0.8,
            }
    )

def plot_sphere(ax, center, radius, color):
    """Draws a 3D sphere surface on the provided Matplotlib axis."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.6, edgecolor='k', linewidth=0.2)

def plot_sphere_test(ax, center, radius, color, angular_position):
    """Draws a slightly more transparent 3D sphere for test rollout visualizations."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=0.2, edgecolor='k', linewidth=0.7)

def plot_box(ax, min_corner, max_corner, color='gray', alpha=0.2):
    """Draws a semi-transparent 3D bounding box to represent the cuboidal enclosure."""
    x_min, y_min, z_min = min_corner
    x_max, y_max, z_max = max_corner
    x = np.array([x_min, x_max])
    y = np.array([y_min, y_max])
    z = np.array([z_min, z_max])

    xx, yy = np.meshgrid(x, y)
    zz_min = np.full_like(xx, z_min)
    zz_max = np.full_like(xx, z_max)
    ax.plot_surface(xx, yy, zz_min, color=color, alpha=alpha, edgecolor='black')
    ax.plot_surface(xx, yy, zz_max, color=color, alpha=alpha, edgecolor='black')

    yy, zz = np.meshgrid(y, z)
    xx_min = np.full_like(yy, x_min)
    xx_max = np.full_like(xx, x_max)
    ax.plot_surface(xx_min, yy, zz, color=color, alpha=alpha, edgecolor='black')
    ax.plot_surface(xx_max, yy, zz, color=color, alpha=alpha, edgecolor='black')

    xx, zz = np.meshgrid(x, z)
    yy_min = np.full_like(xx, y_min)
    yy_max = np.full_like(xx, y_max)
    ax.plot_surface(xx, yy_min, zz, color=color, alpha=alpha, edgecolor='black')
    ax.plot_surface(xx, yy_max, zz, color=color, alpha=alpha, edgecolor='black')

    ax.set_xlabel('X-axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y-axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z-axis', fontsize=12, labelpad=10)

def plot_bottom_wall(ax, z_position, xlim, ylim, color='gray', alpha=0.6, grid=True, checkered=True):
    """Draws a flat plane at a specific Z-height, optionally with a checkered pattern."""
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    zz = np.full_like(xx, z_position)

    if checkered:
        checkers = ((np.indices(xx.shape).sum(axis=0) % 2) == 0).astype(float)
        cmap = ListedColormap(['white', color])
        ax.plot_surface(xx, yy, zz, facecolors=cmap(checkers), alpha=alpha, shade=False, edgecolor='none')
    elif grid:
        ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, edgecolor='k', linewidth=1.5)
    else:
        ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, edgecolor='none')

def plot_snapshot(predicted_node_pos, groundtruth_node_pos, error_pos, error_vel, error_angvel, timestep, plot_folder_path, experiment_name):
    pass

def plot_snapshot_test(predicted_node_pos, predicted_node_angpos, groundtruth_node_pos, groundtruth_node_angpos,
                       predicted_node_vel, predicted_node_angvel, groundtruth_node_vel, groundtruth_node_angvel,
                       error_pos, error_vel, error_angvel,
                       timestep, plot_folder_path, experiment_name, plot_region=(-0.03, 0.03), bottom_wall=False):
    """
    Renders a single frame of the simulation rollout, generating a side-by-side
    comparison of the Ground Truth DEM data and the Model's Prediction.

    Args:
        predicted_node_pos (torch.Tensor): Coordinates predicted by the network.
        groundtruth_node_pos (torch.Tensor): Coordinates from the actual dataset.
        error_pos (float): Mean Absolute Error of position for this timestep.
        timestep (int): The current frame index.
        plot_folder_path (str): Destination directory for the rendered image.
        plot_region (tuple): Boundaries for the visualization axes.
    """
    print("Rendering frame for timestep:", timestep)
    sphere_radius = 0.005 / 2
    margin = 0.001

    fig = plt.figure(figsize=(24, 12))

    # Left Subplot: Ground Truth
    ax1 = fig.add_subplot(121, projection='3d')
    for i,(pos, angpos) in enumerate(zip(groundtruth_node_pos, groundtruth_node_angpos)):
        plot_sphere_test(ax1, np.asarray(pos), sphere_radius, 'grey', np.asarray(angpos))
    if bottom_wall:
        plot_bottom_wall(ax1, z_position=0.0, xlim=(-0.01 - 2*margin, 0.01 + 2*margin), ylim=(-0.01 - 2*margin, 0.01 + 2*margin))

    ax1.set_xlim(plot_region[0] - margin, plot_region[1] + margin)
    ax1.set_ylim(plot_region[0] - margin, plot_region[1] + margin)
    ax1.set_zlim(plot_region[0] - margin, plot_region[1] + margin)
    ax1.set_title('Ground Truth', fontsize=20, fontweight='bold')

    # Right Subplot: Neural Network Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    for i,(pos, angpos) in enumerate(zip(predicted_node_pos, predicted_node_angpos)):
        plot_sphere_test(ax2, np.asarray(pos), sphere_radius, 'b', np.asarray(angpos))
    if bottom_wall:
        plot_bottom_wall(ax2, z_position=0.0, xlim=(-0.01 - 2*margin, 0.01 + 2*margin), ylim=(-0.01 - 2*margin, 0.01 + 2*margin))

    ax2.set_xlim(plot_region[0] - margin, plot_region[1] + margin)
    ax2.set_ylim(plot_region[0] - margin, plot_region[1] + margin)
    ax2.set_zlim(plot_region[0] - margin, plot_region[1] + margin)
    ax2.set_title('Predicted', fontsize=20, fontweight='bold')

    # Formatting and Meta-data labels
    legend_elements = [Patch(facecolor='b', edgecolor='b', label='Predicted'),
                       Patch(facecolor='r', edgecolor='r', label='Ground Truth')]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=16)

    title = f'Time step {timestep}\n'
    title += f'Error (pos): {error_pos:.4f} \n'
    title += f'Error (vel): {error_vel:.4f} \n'
    title += f'Error (angvel): {error_angvel:.4f}'
    fig.suptitle(title, fontsize=24, fontweight='bold')

    plt.tight_layout()

    # Save the frame to disk
    save_path = os.path.join(plot_folder_path, f'frame_{timestep:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def apply_threshold(data, threshold=1e-6):
    """Clamps very small near-zero values to exactly zero for cleaner plotting."""
    return np.where(np.abs(data) < threshold, 0, data)

def plot_momentum_and_energy(predicted_traj, true_traj, system_name, save_folder):
    """Legacy individual plotting function for physics preservation metrics."""
    from utils.utils_dem import calculate_linear_momentum, calculate_angular_momentum, calculate_energy

    set_plot_style()
    num_steps = len(predicted_traj)
    timesteps = range(num_steps)

    dim_size = 3
    p = torch.vstack([calculate_linear_momentum(pred[:, dim_size:2*dim_size]) for pred in predicted_traj])
    l = torch.vstack([calculate_angular_momentum(pred[:, :dim_size], pred[:, dim_size:2*dim_size], pred[:, 2*dim_size:3*dim_size]) for pred in predicted_traj])
    e = torch.tensor([calculate_energy(pred[:, dim_size:2*dim_size], pred[:, 2*dim_size:3*dim_size]) for pred in predicted_traj])

    gt_p = torch.vstack([calculate_linear_momentum(gt[:, dim_size:2*dim_size]) for gt in true_traj])
    gt_l = torch.vstack([calculate_angular_momentum(gt[:, :dim_size], gt[:, dim_size:2*dim_size], gt[:, 2*dim_size:3*dim_size]) for gt in true_traj])
    gt_e = torch.tensor([calculate_energy(gt[:, dim_size:2*dim_size], gt[:, 2*dim_size:3*dim_size]) for gt in true_traj])

    p, l, e = map(lambda x: apply_threshold(x.cpu().numpy()), (p, l, e))
    gt_p, gt_l, gt_e = map(lambda x: apply_threshold(x.cpu().numpy()), (gt_p, gt_l, gt_e))

    fig, axs = plt.subplots(3, 1, figsize=(16, 20))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    data_sets = [(p, gt_p), (l, gt_l), (e[:, np.newaxis], gt_e[:, np.newaxis])]
    titles = ['Linear Momentum', 'Angular Momentum', 'Kinetic Energy']
    y_labels = ['Linear Momentum\nper unit mass (m/s·kg)', 'Angular Momentum\nper unit mass (m²/s·kg)', 'Kinetic Energy\nper unit mass (J/kg)']

    for idx, (ax, (pred_data, gt_data), title, y_label) in enumerate(zip(axs, data_sets, titles, y_labels)):
        components = pred_data.shape[1]
        for i in range(components):
            offset = i * 1e-6
            ax.plot(timesteps, pred_data[:, i] + offset, label=f'Predicted {["X", "Y", "Z"][i] if components > 1 else ""}',
                    color=colors[i], linestyle='-', linewidth=4, alpha=0.4, zorder=3)
            ax.plot(timesteps, gt_data[:, i] + offset, label=f'DEM {["X", "Y", "Z"][i] if components > 1 else ""}',
                    color=colors[i], linestyle='--', linewidth=4, alpha=1.0, zorder=2)

        ax.set_title(f'Total {title} Over Time', fontsize=22, fontweight='bold')
        ax.set_xlabel('Time Step', fontsize=18)
        ax.set_ylabel(y_label, fontsize=18)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    fig.suptitle(f'Conservation of Momentum and Energy - {system_name}', fontsize=24, y=1.02)
    plt.show()

def plot_physics_panel(predicted_traj, true_traj, system_name, save_dir):
    """
    Generates a 3x1 panel comparing physical metrics:
    Kinetic Energy, Linear Momentum (X,Y,Z), and Angular Momentum (X,Y,Z).

    This visually proves whether the GNN has successfully learned to preserve
    the laws of thermodynamics during an extended rollout.
    """
    from utils.utils_dem import calculate_linear_momentum, calculate_angular_momentum, calculate_energy

    os.makedirs(save_dir, exist_ok=True)
    set_plot_style()

    num_steps = len(predicted_traj)
    timesteps = range(num_steps)

    dim_size = 3

    # Fast vectorized extraction of metrics across the entire trajectory sequence
    p = torch.vstack([calculate_linear_momentum(pred[:, dim_size:2*dim_size]) for pred in predicted_traj])
    l = torch.vstack([calculate_angular_momentum(pred[:, :dim_size], pred[:, dim_size:2*dim_size], pred[:, 2*dim_size:3*dim_size]) for pred in predicted_traj])
    e = torch.tensor([calculate_energy(pred[:, dim_size:2*dim_size], pred[:, 2*dim_size:3*dim_size]) for pred in predicted_traj])

    gt_p = torch.vstack([calculate_linear_momentum(gt[:, dim_size:2*dim_size]) for gt in true_traj])
    gt_l = torch.vstack([calculate_angular_momentum(gt[:, :dim_size], gt[:, dim_size:2*dim_size], gt[:, 2*dim_size:3*dim_size]) for gt in true_traj])
    gt_e = torch.tensor([calculate_energy(gt[:, dim_size:2*dim_size], gt[:, 2*dim_size:3*dim_size]) for gt in true_traj])

    # Convert to numpy for matplotlib compatibility
    p, l, e = p.cpu().numpy(), l.cpu().numpy(), e.cpu().numpy()
    gt_p, gt_l, gt_e = gt_p.cpu().numpy(), gt_l.cpu().numpy(), gt_e.cpu().numpy()

    fig, axs = plt.subplots(3, 1, figsize=(16, 18))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue (X), Orange (Y), Green (Z)
    axes_labels = ['X', 'Y', 'Z']
    lw_pred, lw_gt = 3.5, 3.5
    ls_pred, ls_gt = '-', '--'
    alpha_pred, alpha_gt = 0.7, 1.0

    # --- Panel 1: Kinetic Energy ---
    axs[0].plot(timesteps, e, color='purple', linewidth=lw_pred, linestyle=ls_pred, alpha=alpha_pred, label='Predicted Total KE')
    axs[0].plot(timesteps, gt_e, color='black', linewidth=lw_gt, linestyle=ls_gt, alpha=alpha_gt, label='DEM Total KE')
    axs[0].set_title('Total Kinetic Energy (per Unit Mass)', fontweight='bold')
    axs[0].set_ylabel('Kinetic Energy (J/kg)')
    axs[0].legend(loc='upper right', fontsize=14)

    # --- Panel 2: Linear Momentum (X, Y, Z) ---
    for i in range(3):
        # Adding a microscopic offset prevents lines from completely obscuring each other if perfectly identical
        offset = i * 1e-6
        axs[1].plot(timesteps, p[:, i] + offset, color=colors[i], linewidth=lw_pred, linestyle=ls_pred, alpha=alpha_pred, label=f'Pred {axes_labels[i]}')
        axs[1].plot(timesteps, gt_p[:, i] + offset, color=colors[i], linewidth=lw_gt, linestyle=ls_gt, alpha=alpha_gt, label=f'DEM {axes_labels[i]}')

    axs[1].set_title('Total Linear Momentum (per Unit Mass)', fontweight='bold')
    axs[1].set_ylabel('Linear Momentum (m/s)')
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

    # --- Panel 3: Angular Momentum (X, Y, Z) ---
    for i in range(3):
        offset = i * 1e-8
        axs[2].plot(timesteps, l[:, i] + offset, color=colors[i], linewidth=lw_pred, linestyle=ls_pred, alpha=alpha_pred, label=f'Pred {axes_labels[i]}')
        axs[2].plot(timesteps, gt_l[:, i] + offset, color=colors[i], linewidth=lw_gt, linestyle=ls_gt, alpha=alpha_gt, label=f'DEM {axes_labels[i]}')

    axs[2].set_title('Total Angular Momentum (per Unit Mass)', fontweight='bold')
    axs[2].set_ylabel('Angular Momentum (m²/s)')
    axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)

    # Global Formatting adjustments
    for ax in axs:
        ax.set_xlabel('Time Step')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        if y_range > 0:
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'physics_panel_{system_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_angular_velocity_components(pred_angular_velocity, gt_angular_velocity, save_folder, model_name="DynSolver"):
    """
    Plots the final post-collision angular velocity components across different impact angles.
    Used exclusively for the Oblique Wall Collision Benchmark.
    """
    os.makedirs(save_folder, exist_ok=True)
    set_plot_style()

    # Extract components and angles from the aggregated arrays
    angles = pred_angular_velocity[:, 0]
    pred_x, pred_y, pred_z = pred_angular_velocity[:, 1], pred_angular_velocity[:, 2], pred_angular_velocity[:, 3]
    gt_x, gt_y, gt_z = gt_angular_velocity[:, 1], gt_angular_velocity[:, 2], gt_angular_velocity[:, 3]

    pred_magnitude = np.sqrt(pred_x**2 + pred_y**2 + pred_z**2)
    gt_magnitude = np.sqrt(gt_x**2 + gt_y**2 + gt_z**2)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # --- Plot Components (X, Y, Z) individually ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))

    axs[0].plot(angles, pred_x, label=model_name, linestyle='-', linewidth=3, marker='o', markersize=8, color=colors[0])
    axs[0].plot(angles, gt_x, label='DEM Ground Truth', linestyle='--', linewidth=3, marker='x', markersize=8, color=colors[1])
    axs[0].set_title('Angular Velocity - X Component', fontweight='bold')
    axs[0].set_ylabel('X Ang Vel (rad/s)')

    axs[1].plot(angles, pred_y, label=model_name, linestyle='-', linewidth=3, marker='o', markersize=8, color=colors[0])
    axs[1].plot(angles, gt_y, label='DEM Ground Truth', linestyle='--', linewidth=3, marker='x', markersize=8, color=colors[1])
    axs[1].set_title('Angular Velocity - Y Component', fontweight='bold')
    axs[1].set_ylabel('Y Ang Vel (rad/s)')

    axs[2].plot(angles, pred_z, label=model_name, linestyle='-', linewidth=3, marker='o', markersize=8, color=colors[0])
    axs[2].plot(angles, gt_z, label='DEM Ground Truth', linestyle='--', linewidth=3, marker='x', markersize=8, color=colors[1])
    axs[2].set_title('Angular Velocity - Z Component', fontweight='bold')
    axs[2].set_ylabel('Z Ang Vel (rad/s)')

    for ax in axs:
        ax.set_xlabel('Angle of Impact (degrees)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'oblique_angular_velocity_components.png'), bbox_inches='tight')
    plt.close(fig)

    # --- Plot Total Magnitude ---
    fig_mag = plt.figure(figsize=(10, 6))
    plt.plot(angles, pred_magnitude, label=model_name, linestyle='-', linewidth=3, marker='o', markersize=8, color=colors[0])
    plt.plot(angles, gt_magnitude, label='DEM Ground Truth', linestyle='--', linewidth=3, marker='x', markersize=8, color=colors[1])
    plt.title('Resulting Angular Velocity vs Angle of Impact', fontweight='bold')
    plt.xlabel('Angle of Impact (degrees)')
    plt.ylabel('Magnitude (rad/s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'oblique_angular_velocity_magnitude.png'), bbox_inches='tight')
    plt.close(fig_mag)

def create_gif(folder='imageio_temp_files', save_folder='.', gif_name='trajectory.gif'):
    """
    Compiles a sequence of individually rendered PNG frames into an animated GIF.

    Args:
        folder (str): Directory containing the target PNG files.
        save_folder (str): Destination directory for the compiled GIF.
        gif_name (str): Filename for the compiled GIF.
    """
    images = []

    # Define a sorting key to ensure frames are stitched in correct numerical order
    # Extracts the integer '0001' from a filename string like 'frame_0001.png'
    def extract_step(file_name):
        try:
            return int(file_name.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            return -1

    valid_files = [f for f in os.listdir(folder) if f.startswith('frame_') and f.endswith('.png')]
    sorted_files = sorted(valid_files, key=extract_step)

    # Read files from disk sequentially
    for file_name in sorted_files:
        file_path = os.path.join(folder, file_name)
        images.append(imageio.imread(file_path))

    save_dir = os.path.join('.', save_folder)
    os.makedirs(save_dir, exist_ok=True)
    save_gif_path = os.path.join(save_dir, gif_name)

    if images:
        # duration is milliseconds per frame in imageio 2.x; loop=0 repeats forever.
        imageio.mimsave(save_gif_path, images, duration=FRAME_DURATION_MS, loop=0)
    else:
        print(f"Warning: No valid frame images found in {folder} to create GIF.")
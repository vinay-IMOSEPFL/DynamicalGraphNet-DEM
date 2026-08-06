# Dynami-CAL GraphNet: Discrete Element Method (DEM) Dynamics

This repository contains the standalone PyTorch implementation of the Discrete Element Method (DEM) experiments for the paper:

> **A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems**
> Sharma & Fink — *Nature Communications*, 2026
> https://www.nature.com/articles/s41467-025-67802-5

The full codebase — covering human motion, molecular dynamics, and N-body experiments — is available in the [main repository]. This repository specifically isolates the complex 3D granular mechanics experiments, separating them into two distinct pipelines: **Case 04** (Simple / Homogeneous) and **Case 05** (Hard / Heterogeneous).

The data is available on Zenodo at https://zenodo.org/records/19691595 and is automatically downloaded by the `get_data.py` scripts in each case folder.

---

## Repository Structure

```
DYNAMI-CAL-DEM/
├── model/
│   └── model_dem.py                  # Core DynamicsSolver GNN architecture
├── utils/
│   ├── trainer_dem.py                # Training loop, gradient accumulation, checkpointing
│   └── utils_dem.py                  # MLP builder, momentum and energy calculation helpers
│
├── case_04_dem_simple/               # Homogeneous interaction cases
│   ├── config.py                     # Hyperparameters, geometry, directory paths
│   ├── boundary_model.py             # Ghost-node cuboid reflections (SphereWallInteraction)
│   ├── dataset.py                    # PyG Dataset loader with split and case-specific modes
│   ├── get_data.py                   # Zenodo downloader: homogeneous + benchmark datasets
│   ├── preprocess.py                 # CSV → PyG graph converter (sliding-window, t-1/t/t+1)
│   ├── rollout_evaluator.py          # Single-timestep autoregressive rollout evaluator
│   └── visualization.py              # 3D scatter plots, physics conservation panels, GIF builder
│
├── case_05_dem_hard/                 # Heterogeneous gravity and cylinder cases
│   ├── config.py
│   ├── boundary_model.py             # Cuboid AND rotating cylinder boundary models
│   ├── dataset.py
│   ├── get_data.py                   # Zenodo downloader: gravity cuboid + rotating cylinder
│   ├── preprocess.py                 # Processes both cuboid and cylinder with correct boundaries
│   ├── rollout_evaluator.py          # Dual-timestep micro-stepping rollout evaluator
│   ├── render_frames.py              # Standalone multiprocessed renderer for saved .pt snapshots
│   └── visualization.py
│
├── main_dem_simple.py                # Execution pipeline for Case 04
├── main_dem_hard.py                  # Execution pipeline for Case 05
│
├── environment.yml                   # Pinned Conda environment specification
└── requirements.txt                  # Pinned pip package list
```

---

## Installation

The full software environment is pinned in two equivalent specifications — use whichever fits your workflow.

**Conda**

```bash
conda env create -f environment.yml
conda activate dem-dyngnet
```

**pip**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Environment details

The pinned versions below are the exact ones the Case 04 and Case 05 pipelines were verified against end to end — download, preprocess, train, test, benchmark and render — on macOS, CPU.

| Package | Version |
| --- | --- |
| Python | 3.12 |
| torch | 2.7.0 |
| torch-geometric | 2.6.1 |
| numpy | 1.26.4 |
| pandas | 2.2.2 |
| matplotlib | 3.9.2 |
| seaborn | 0.13.2 |
| imageio | 2.33.1 |
| tqdm | 4.66.5 |

Three things worth knowing about these choices:

- **PyTorch version matters.** Preprocessed graphs are PyTorch Geometric `Data` objects and are loaded with `weights_only=False`. That argument is required from PyTorch 2.6 onwards, where the `torch.load` default flipped to `weights_only=True`; without it, dataset loading fails outright.
- **The compiled PyG extensions are not needed.** Only `Data` and `DataLoader` are used, so `torch-scatter`, `torch-sparse` and `torch-cluster` — by far the most awkward packages to install — can be skipped entirely.
- **GPU users** should install the matching CUDA build of PyTorch first, then the remaining requirements:

  ```bash
  pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126
  pip install -r requirements.txt
  ```

---

## Step 1: Downloading and Preprocessing Data

Each case includes self-contained scripts to fetch raw simulation data from Zenodo and convert it into PyTorch Geometric (`.pt`) graph sequences. This step only needs to be done once.

### Fetch Raw Data

```bash
# Download Case 04: homogeneous 60-sphere dataset + oblique benchmarks
python case_04_dem_simple/get_data.py

# Download Case 05: heterogeneous gravity cuboid + 2073-sphere rotating cylinder
python case_05_dem_hard/get_data.py
```

Both scripts download with a progress bar, skip files already present, and delete `.zip` archives after extraction. The Case 05 script accepts `--keep-zip` to retain the archives; the Case 04 script always removes them.

**Case 04 downloads and organises into:**
```
case_04_dem_simple/data/
├── homogeneous/
│   ├── training/        case_01 … case_05
│   ├── validation/      case_06
│   ├── extrapolation/   case_07
│   └── benchmark/
│       ├── oblique_wall_collisions/   10_deg, 30_deg, 45_deg, 60_deg, 90_deg
│       └── oblique_sphere_collisions/ data_at_timestep_000.csv … (100 files)
```

**Case 05 downloads and organises into:**
```
case_05_dem_hard/data/
└── heterogeneous/gravity/
    ├── training/        case_01 … case_05
    ├── validation/      case_06
    ├── extrapolation/   case_07
    └── rotating_cylinder/ data_at_timestep_000.csv … (2000 files)
```

### Preprocess into Graphs

The preprocessors crawl the raw data directories, build PyTorch Geometric graphs using a sliding (t−1, t, t+1) window, and serialize each case as a `graph_list.pt` file. Graph topology is computed by the boundary interaction model: real sphere positions are reflected across the walls to generate ghost nodes, and edges are established for all node pairs within the interaction threshold (1.25 × sphere diameter).

```bash
# Process Case 04
python case_04_dem_simple/preprocess.py

# Process Case 05
python case_05_dem_hard/preprocess.py
```

The Case 04 preprocessor uses three distinct boundary configurations: standard cuboid walls for the main training/validation/extrapolation splits, an effectively infinite bounding box for the sphere-collision benchmarks, and a single floor plane at z = 0 for the wall-collision benchmarks. The Case 05 preprocessor uses the standard cuboid for gravity training cases and a cylindrical boundary model for the rotating drum.

---

## A Note on Checkpoints

No pretrained weights ship with this repository, so every test and benchmark mode below requires that you first run the corresponding `--mode train` and produce `saved_models/model_checkpoint_best_val.pth`.

If that checkpoint is missing, evaluation does **not** stop — it proceeds with randomly initialized weights and still writes plausible-looking GIFs and physics panels. Case 04 prints a warning in this situation; Case 05 does not. Confirm that the run logs `Loaded best validation model from ...` before treating any output as a result.

---

## Step 2: Case 04 — Simple / Homogeneous

This case focuses on 60 homogeneous spheres confined inside a stationary cuboidal enclosure (0.03 m × 0.03 m × 0.03 m). All sphere–sphere and sphere–wall interactions are of the same type, with no external forces. The model is trained at a timestep of 1×10⁻⁴ s.

**Key hyperparameters** (from `case_04_dem_simple/config.py`): learning rate 3×10⁻⁵, batch size 64, 500 epochs, latent size 128, 5 message-passing rounds.

All Case 04 experiments are run through `main_dem_simple.py` using the `--mode` flag.

### Train

```bash
python main_dem_simple.py --mode train
```

Trains on cases 01–05, validates on case 06 every 5 epochs, and saves the best checkpoint to `case_04_dem_simple/saved_models/`.

### Test (Long-Horizon Rollout)

Evaluates the trained model on case 07 autoregressively — the model's own predictions are fed back as input at each step. Outputs a `.gif` animation of the rollout and physics conservation plots tracking total energy, linear momentum, and angular momentum over time.

```bash
python main_dem_simple.py --mode test --plot --save_data
```

`--save_data` serializes the full kinematic state at each rollout step as individual `.pt` snapshot files under `results/`. Without it, only the GIF is retained after the run. `--plot` enables frame rendering and GIF generation.

### Benchmark: Oblique Sphere Collisions

Evaluates the model on isolated two-sphere oblique and head-on collisions inside an effectively infinite bounding box. Tests whether the model correctly transfers linear and angular momentum across a range of impact angles.

```bash
python main_dem_simple.py --mode benchmark_sphere_collisions --plot
```

### Benchmark: Oblique Wall Collisions

Evaluates a single sphere impacting a flat wall at five precise angles: 10°, 30°, 45°, 60°, and 90°. Checks that the model correctly decomposes and applies the normal and tangential impulse components.

```bash
python main_dem_simple.py --mode benchmark_wall_collisions --plot
```

---

## Step 3: Case 05 — Hard / Heterogeneous

This pipeline introduces external forces (gravity) and significantly more complex boundary conditions, including a rotating cylindrical drum. Sphere properties (radius, density/material) vary across particles, which complicates the interaction structure. The gravity model trains at 1×10⁻⁴ s and the cylinder dataset is saved at 1×10⁻³ s.

**Key hyperparameters** (from `case_05_dem_hard/config.py`): learning rate 3×10⁻⁴, batch size 64, 500 epochs, latent size 128, 5 message-passing rounds, external force MLP enabled.

All Case 05 experiments are run through `main_dem_hard.py`.

### Train (Gravity Enabled)

Trains on 60 heterogeneous spheres falling under gravity inside a cuboidal box (cases 01–05). The external force MLP (`use_ext_force=True`) is automatically enabled to handle gravity as an additional per-node input. Validates on case 06 every 5 epochs.

```bash
python main_dem_hard.py --mode train
```

### Test: Gravity Cuboid Rollout

Runs a long-horizon autoregressive rollout with gravity enabled and produces a GIF plus a physics conservation panel.

Note that this mode currently evaluates **case 06** — the same case used for validation during training — not the held-out extrapolation case 07. Case 07 is downloaded and preprocessed by the Case 05 pipeline but is not evaluated by any mode. Change `case_name` in `main_dem_hard.py` to `"case_07"` for a genuinely held-out evaluation.

```bash
python main_dem_hard.py --mode test --plot
```

### Test: Massive Extrapolation to a Rotating Cylinder

Evaluates the model zero-shot on a 2,073-sphere system tumbling inside a rotating cylindrical drum — a geometry and scale the model has never encountered during training.

**Note on temporal synchronization:** The GNN solver operates at a fine internal timestep of 1×10⁻⁴ s to maintain numerical stability, while the reference dataset is saved at 1×10⁻³ s. The `evaluate_rollout` function bridges this gap by automatically performing 10 internal micro-steps before checking its state against the next available ground-truth frame. The number of micro-steps is computed directly from the ratio of `SAMPLE_TIME_STEP_CYLINDER / SAMPLE_TIME_STEP_CUBOID` defined in `config.py`.

```bash
python main_dem_hard.py --mode cylinder --plot --save_data
```

As with Case 04, `--save_data` is optional. Without it only the GIF is retained. With it, individual `.pt` snapshot files are preserved under `results/` for downstream analysis or re-rendering.

---

## Standalone Frame Renderer

Case 05 includes a `render_frames.py` script for re-rendering saved `.pt` rollout snapshots without re-running the physics. This is useful for regenerating plots at a different frame frequency, changing visualization parameters, or recovering visualizations after a run where `--plot` was not passed. It requires snapshots produced by a `--save_data` run.

Paths are resolved relative to the current working directory, and each case writes its results under its own folder — so from the repository root the data directory is `case_05_dem_hard/results/...`:

```bash
# Re-render a saved cuboid rollout
python case_05_dem_hard/render_frames.py \
    --data_dir case_05_dem_hard/results/case_06_gravity_rollout/rollout_data \
    --frequency 10

# Re-render a saved cylinder rollout
python case_05_dem_hard/render_frames.py \
    --data_dir case_05_dem_hard/results/rotating_cylinder_rollout/rollout_data \
    --cylinder \
    --frequency 25
```

The renderer dispatches frame generation across all available CPU cores using `ProcessPoolExecutor`, compiles a GIF from the rendered frames, and saves everything alongside the original data directory. It renders frames and builds the GIF only; it does not produce a physics conservation panel.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dyngnet2026,
  title   = {A physics-informed graph neural network conserving linear and angular momentum for dynamical systems},
  author  = {Sharma, Vinay and Fink, Olga},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.1038/s41467-025-67802-5},
  url     = {https://www.nature.com/articles/s41467-025-67802-5}
}
```
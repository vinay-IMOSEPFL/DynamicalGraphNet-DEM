# Dynami-CAL GraphNet: Discrete Element Method (DEM) Dynamics

This repository contains the standalone PyTorch implementation of the Discrete Element Method (DEM) experiments for the paper:

> **A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems**
> Sharma & Fink — *Nature Communications*, 2026
> https://www.nature.com/articles/s41467-025-67802-5

The full codebase — covering human motion, molecular dynamics, and N-body experiments — is available in the main repository. This repository isolates the 3D granular mechanics experiments as two pipelines: **Case 04** (homogeneous) and **Case 05** (heterogeneous, under gravity).

Data is hosted on Zenodo at https://zenodo.org/records/19691595 and is downloaded automatically by the `get_data.py` script in each case folder.

---

## Repository Structure

```
DynamicalGraphNet-DEM/
├── model/
│   └── model_dem.py                  # DynamicsSolver GNN architecture
├── utils/
│   ├── trainer_dem.py                # Training loop, gradient accumulation, checkpointing
│   ├── utils_dem.py                  # MLP builder, momentum and energy helpers
│   └── metrics_dump.py               # Serialises rollout errors to JSON/CSV
│
├── case_04_dem_simple/               # Homogeneous interaction cases
│   ├── config.py                     # Hyperparameters, geometry, directory paths
│   ├── boundary_model.py             # Ghost-node cuboid reflections
│   ├── dataset.py                    # PyG dataset loader
│   ├── get_data.py                   # Zenodo downloader
│   ├── preprocess.py                 # CSV to PyG graphs, sliding (t-1, t, t+1) window
│   ├── rollout_evaluator.py          # Single-timestep autoregressive rollout
│   └── visualization.py              # 3D frames, physics panels, GIF builder
│
├── case_05_dem_hard/                 # Heterogeneous gravity and cylinder cases
│   ├── config.py
│   ├── boundary_model.py             # Cuboid and rotating cylinder boundaries
│   ├── dataset.py
│   ├── get_data.py
│   ├── preprocess.py
│   ├── rollout_evaluator.py          # Dual-timestep micro-stepping rollout
│   ├── render_frames.py              # Standalone renderer for saved .pt snapshots
│   └── visualization.py
│
├── scripts/
│   ├── run_training.sh               # Detached training launcher with timestamped logs
│   ├── run_eval.sh                   # Detached evaluation launcher
│   ├── collect_metrics.py            # Aggregates all metrics.json into one table
│   └── parse_training_log.py         # Loss and validation curves from a training log
│
├── main_dem_simple.py                # Entry point for Case 04
├── main_dem_hard.py                  # Entry point for Case 05
│
├── reports/                          # Metrics summary and training curves
├── decisions.md                      # Reproduction log: measurements, deviations, findings
├── environment.yml                   # Pinned Conda environment
└── requirements.txt                  # Pinned pip requirements
```

---

## Installation

The environment is pinned in two equivalent specifications.

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

These versions were verified end to end — download, preprocess, train, evaluate, benchmark, render — on macOS (CPU) and on Linux with an NVIDIA RTX 2080 Ti.

Three notes on these choices:

- **PyTorch version matters.** Preprocessed graphs are PyTorch Geometric `Data` objects loaded with `weights_only=False`. That argument is required from PyTorch 2.6 onwards, where the `torch.load` default flipped to `weights_only=True`; without it, dataset loading fails outright.
- **The compiled PyG extensions are not needed.** Only `Data` and `DataLoader` are used, so `torch-scatter`, `torch-sparse` and `torch-cluster` can be skipped.
- **No separate step is needed for CUDA.** The default PyPI wheel for `linux-x86_64` is already the CUDA build (torch 2.7.0 ships CUDA 12.6), so `pip install -r requirements.txt` yields a GPU-capable environment as-is. Use `--index-url` only to pin a different CUDA version or to force a CPU-only build on Linux. The macOS wheel is CPU-only.

Everything installs through pip rather than conda deliberately. Mixing conda-forge numpy with the pip PyTorch wheel places two OpenMP runtimes in one process, which aborts the interpreter on import in an order-dependent way. See `decisions.md` §2.3.

Verify the GPU is actually usable before starting a long run:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 1: Download and preprocess the data

Each case fetches its raw simulation data from Zenodo and converts it to PyTorch Geometric graph sequences. This is a one-time step.

```bash
python case_04_dem_simple/get_data.py
python case_05_dem_hard/get_data.py
```

Both scripts show a progress bar, skip files already present, and delete `.zip` archives after extraction. Case 05 accepts `--keep-zip` to retain the archives; Case 04 always removes them.

**Case 04 layout:**
```
case_04_dem_simple/data/homogeneous/
├── training/        case_01 … case_05
├── validation/      case_06
├── extrapolation/   case_07
└── benchmark/
    ├── oblique_wall_collisions/    10_deg, 30_deg, 45_deg, 60_deg, 90_deg
    └── oblique_sphere_collisions/  101 timestep files
```

**Case 05 layout:**
```
case_05_dem_hard/data/heterogeneous/gravity/
├── training/          case_01 … case_05
├── validation/        case_06
├── extrapolation/     case_07
└── rotating_cylinder/ 2001 timestep files
```

Then build the graphs:

```bash
python case_04_dem_simple/preprocess.py
python case_05_dem_hard/preprocess.py
```

Graph topology comes from the boundary interaction model: real sphere positions are reflected across the walls to create ghost nodes, and edges connect all node pairs within the interaction threshold (1.25 × sphere diameter). The sliding (t−1, t, t+1) window drops the first and last frame of each sequence, so a 1501-frame case yields 1499 graphs.

Expected output:

| Split | Graphs |
| --- | --- |
| Case 04 and Case 05 cuboid cases (`case_01` … `case_07`) | 1499 each |
| Case 04 `oblique_sphere_collisions` | 99 |
| Case 04 `oblique_wall_collisions/{10,30,45,60,90}_deg` | 199 each |
| Case 05 `rotating_cylinder` | 1999 (8,292 nodes each) |

Preprocessing is single-threaded and CPU-bound; the cylinder dominates. Budget roughly 12 minutes for Case 04 and 25 minutes for Case 05, and around 3 GB of disk for the two `dataset/` directories.

---

## Checkpoints

Trained checkpoints for both cases ship with this repository:

```
case_04_dem_simple/saved_models/model_checkpoint_best_val.pth
case_05_dem_hard/saved_models/model_checkpoint_best_val.pth
```

They are the ones the metrics in `reports/metrics_summary.md` were measured with, so the evaluation modes below can be run directly without training first.

If you retrain, note that a **missing** checkpoint does not stop evaluation — the pipeline proceeds with randomly initialized weights and still writes plausible-looking GIFs and physics panels. Every evaluation mode prints either

```
Loaded best validation model from .../model_checkpoint_best_val.pth
```

or an explicit warning. Confirm the former appears before treating any output as a result; `metrics.json` also records a `checkpoint_loaded` flag for each run.

---

## Step 2: Case 04 — homogeneous

60 identical spheres inside a stationary 0.03 m cuboidal enclosure, no external forces, model timestep 1×10⁻⁴ s.

**Hyperparameters** (`case_04_dem_simple/config.py`): learning rate 3×10⁻⁴, batch size 64, 200 epochs, latent size 128, 5 message-passing rounds, Adam.

```bash
# Train on cases 01-05, validating on case 06 every 5 epochs
python main_dem_simple.py --mode train

# Autoregressive rollout on the held-out case 07
python main_dem_simple.py --mode test --plot --save_data

# Two-sphere oblique collisions in free space
python main_dem_simple.py --mode benchmark_sphere_collisions --plot

# Single-sphere wall impacts at 10, 30, 45, 60 and 90 degrees
python main_dem_simple.py --mode benchmark_wall_collisions --plot
```

`--plot` enables frame rendering and GIF generation. `--save_data` additionally keeps the per-step `.pt` snapshots under `results/` for later re-rendering or analysis. `--save_plot` retains the individual PNG frames after the GIF is built.

Checkpoint selection uses a 200-step autoregressive rollout on the validation case, scored by the sum of the scaled position, velocity and angular-velocity errors.

## Step 3: Case 05 — heterogeneous, under gravity

Adds gravity via an external-force MLP (`use_ext_force=True`) and a rotating cylindrical boundary. The gravity cuboid trains at 1×10⁻⁴ s; the cylinder ground truth is saved at 1×10⁻³ s.

**Hyperparameters** (`case_05_dem_hard/config.py`): learning rate 3×10⁻⁴, batch size 64, 200 epochs, latent size 128, 5 message-passing rounds, Adam, external-force MLP enabled.

```bash
# Train on gravity cases 01-05, validating on case 06 every 5 epochs
python main_dem_hard.py --mode train

# Autoregressive rollout on the cuboid case 06
python main_dem_hard.py --mode test --plot --save_data

# Zero-shot extrapolation to the 2,073-sphere rotating cylinder
python main_dem_hard.py --mode cylinder --plot --save_data
```

**Temporal synchronization.** The solver integrates at 1×10⁻⁴ s while the cylinder reference data is stored at 1×10⁻³ s, so `evaluate_rollout` takes 10 internal micro-steps before comparing against the next ground-truth frame. The count is derived from `SAMPLE_TIME_STEP_CYLINDER / SAMPLE_TIME_STEP_CUBOID` in `config.py`; for the cuboid modes the ratio is 1 and the rollout advances one model step per frame.

The cylinder run is the expensive one: 1,998 sync points × 10 micro-steps ≈ 20,000 forward passes over an 8,292-node graph, with a full pairwise distance computation and graph rebuild at every micro-step. Expect roughly 20 minutes on an RTX 2080 Ti and about 400 MB of snapshots with `--save_data`.

---

## Metrics

Every evaluation writes `metrics.json` and `per_step_errors.csv` into a `metrics/` folder beside its results, recording mean and final scaled MAE for position, velocity and angular velocity, the rollout length, wall-clock time, and whether the checkpoint loaded.

Aggregate them into one table:

```bash
python scripts/collect_metrics.py          # -> reports/metrics_summary.{md,csv,json}
```

Plot loss and validation curves from a training log:

```bash
python scripts/parse_training_log.py logs/train_case04_<stamp>.log --tag case04
```

Errors are dimensionless, divided by training-set maxima: `edge_feat_max` for position (6.25×10⁻³ m, the interaction cutoff — not the box size), `node_vel_max` for velocity, `node_angvel_max` for angular velocity.

For long unattended runs, `scripts/run_training.sh` and `scripts/run_eval.sh` launch a case detached under `setsid` with timestamped logs, so the run survives the controlling terminal exiting:

```bash
setsid nohup ./scripts/run_training.sh case04 0 &   # case, GPU index
```

---

## Standalone frame renderer

Case 05 includes `render_frames.py` for re-rendering saved `.pt` snapshots without re-running the physics — useful for changing frame frequency or recovering visualizations from a run where `--plot` was omitted. It requires snapshots from a `--save_data` run. There is no equivalent script for Case 04.

Paths resolve relative to the working directory, so from the repository root the data directory is `case_05_dem_hard/results/...`:

```bash
python case_05_dem_hard/render_frames.py \
    --data_dir case_05_dem_hard/results/case_06_gravity_rollout/rollout_data \
    --frequency 10

python case_05_dem_hard/render_frames.py \
    --data_dir case_05_dem_hard/results/rotating_cylinder_rollout/rollout_data \
    --cylinder \
    --frequency 25
```

It renders frames across available CPU cores with `ProcessPoolExecutor` and compiles a GIF. It does not produce a physics conservation panel.

---

## Reproduction notes

`decisions.md` is a full log of the reproduction: measured timings, hardware, every deviation and its rationale, and the defects found and fixed in the original code. Points worth knowing before starting:

- **Training benefits from a GPU; rollouts do not.** Training epochs run about 4× faster on an RTX 2080 Ti than on a 14-core Apple M4 Pro, but the 200-step validation rollout is roughly 1.5× *slower* on the GPU. The rollout is sequential over a ~420-node graph, so kernel-launch latency dominates. The cylinder is the one inference workload large enough to benefit.
- **Case 05's `--mode test` evaluates `case_06`**, which is also the validation case used for checkpoint selection. It is therefore not a held-out generalization result. Case 04's `--mode test` uses the genuinely held-out `case_07`.
- **Spheres can leave the enclosure during long rollouts.** A wall ghost is a mirror image, so the sphere-to-ghost separation is twice the sphere's distance to the wall. Tested against the same `THRESHOLD` as sphere-sphere pairs, this gives the wall half the interaction range of a particle (0.625 mm of surface gap versus 1.25 mm). A fast sphere can cross the band before the model reverses it, and beyond 3.125 mm past the plane no wall edge exists, so nothing pulls it back. In a 1499-step Case 04 rollout roughly a quarter of the spheres end up outside the box, drifting ballistically. `decisions.md` §21 has the measurements and the candidate fixes.
- **Rendering spawns one process per CPU core.** On a host with many cores and limited memory, cap `max_workers` in `rollout_evaluator.py`.
- **Results are single-seed.** Runs are seeded (`set_seed(100)`) but no seed-variance study was performed, so no error bars accompany the reported numbers.

---

## Citation

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

# Decisions & Findings Log

Working notes captured while auditing, fixing and reproducing this repository from
scratch. This is the raw material for the final `readme.md` — everything here is
either a change that was made, a constraint that was discovered, or a number that
was measured.

**Reproduction machine:** Apple M4 Pro, 14 cores, 48 GB RAM, macOS. **No CUDA** —
all training and evaluation is CPU-only.

---

## 1. Code defects found and fixed

Three defects made a fresh clone unusable. All three were confirmed by running the
pipelines, not by reading alone.

### 1.1 `torch.load` fails on PyTorch >= 2.6 (blocker)

**Where:** `case_04_dem_simple/dataset.py`, `case_05_dem_hard/dataset.py`

PyTorch 2.6 flipped the `torch.load` default to `weights_only=True`, which refuses
to unpickle PyTorch Geometric `Data` objects:

```
_pickle.UnpicklingError: Weights only load failed.
  Unsupported global: GLOBAL torch_geometric.data.data.DataEdgeAttr
```

Every mode of both pipelines died on the first dataset load. The README pinned no
versions, so a new user installing current PyTorch hit this immediately.

**Fix:** pass `weights_only=False` at all four call sites, with a comment
explaining why. Model-checkpoint loads were deliberately left untouched — those are
plain state dicts and are safe under the strict default.

### 1.2 `.cpu()` called on a numpy array (blocker)

**Where:** `case_04_dem_simple/visualization.py`, `case_05_dem_hard/visualization.py`,
inside `plot_snapshot_test`.

The render workers already convert to numpy (`data['pred_pos'].numpy()`) before
calling the plotter, but the plotter then calls `pos.cpu().numpy()`:

```
AttributeError: 'numpy.ndarray' object has no attribute 'cpu'
```

Version-independent. Broke `--plot` for every cuboid rollout in both cases. The
cylinder renderer was already correct — it guards with `torch.is_tensor`.

**Fix:** `np.asarray(pos)`, which accepts tensors and arrays alike.

### 1.3 `--plot --save_data` crashed the physics panel (blocker)

**Where:** both `rollout_evaluator.py` files, plus the `main_*.py` call sites.

The evaluators only accumulated trajectories `if not save_data`, but `main` then fed
those lists to `plot_physics_panel`:

```
RuntimeError: vstack expects a non-empty TensorList
```

This was the README's own documented command
(`--mode test --plot --save_data`).

**Fix considered and rejected:** delete the `if not save_data` guard. That would
hold every rollout state in RAM, defeating the author's evident intent for the
2073-sphere cylinder run.

**Fix applied:** a "Stage 2b" reconstruction step rebuilds the trajectory lists from
the snapshots already written to disk. Peak memory during the rollout is unchanged
and the return value is now correct in both modes.

**Verified:** the physics panel rendered via the disk-rebuilt path is byte-for-byte
identical to the one rendered from the in-memory path.

---

## 2. Environment: what actually works

### 2.1 The real dependency list

`torchvision` was in the README's install command but **is never imported anywhere**
in the repo. Dropped.

The compiled PyG extensions — `torch-scatter`, `torch-sparse`, `torch-cluster` —
are **not required**. The code only uses `Data`, `DataLoader` and plain torch ops
(`index_add_`, `cdist`). These are the most painful packages to install in a PyG
project, so omitting them materially lowers the barrier to reproduction.

Actual third-party imports: `torch`, `torch_geometric`, `numpy`, `pandas`,
`matplotlib`, `seaborn`, `tqdm`, `imageio`.

### 2.2 Pinned versions (verified)

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

All eight exist on PyPI (HTTP 200 checked).

### 2.3 The OpenMP trap (important, non-obvious)

The **first** `environment.yml` drafted here installed the scientific stack from
conda-forge and only torch from pip. It solved cleanly, all pins resolved, and it
looked correct — but the resulting environment **aborted on import**:

```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
Abort trap: 6
```

Cause: conda-forge numpy pulls in conda's `libomp` via OpenBLAS, while the pip
PyTorch wheel bundles its own `libomp`. Two OpenMP runtimes in one process.

The trap is that the failure is **import-order dependent**:

- `import numpy, torch` -> survives
- `import torch, numpy` -> aborts

`main_dem_simple.py` and `main_dem_hard.py` both import torch before numpy, so this
environment would have broken the actual pipelines while passing a naive smoke test.

**Rejected workaround:** `KMP_DUPLICATE_LIB_OK=TRUE`. It is documented as possibly
producing silently incorrect results — unacceptable in an environment whose entire
purpose is reproducibility.

**Fix applied:** conda supplies only Python and pip; the whole stack comes from pip
via `-r requirements.txt`. That leaves exactly one `libomp` (torch's). Both import
orders now succeed and every repo module imports cleanly.

Side benefit: `environment.yml` and `requirements.txt` can no longer drift, because
the former literally references the latter.

**Lesson for the README:** a "the solver resolved it" check is not a verification.
The environment must be exercised by importing in the order the code actually uses.

---

## 3. Documentation corrections made

| Claim in original README | Reality |
| --- | --- |
| `case_04_dem_simple/render_frames.py` listed in the file tree | File does not exist; only Case 05 has one |
| Hard `--mode test` evaluates "held-out extrapolation case 07" | Code evaluates `case_06`, the validation case |
| Renderer example `--data_dir results/...` | Results live under `case_05_dem_hard/results/...`; literal command errors out |
| "Both scripts ... pass `--keep-zip`" | Only Case 05 parses it; Case 04 hardcodes `kz = False` and ignores argv |
| `pip install torch torchvision` | `torchvision` never imported |
| Renderer "saves everything" | It builds frames + GIF only; never produces a physics panel despite importing one |

Also added a **"A Note on Checkpoints"** section: no pretrained weights ship with
the repo, and a missing checkpoint does **not** halt evaluation — it silently
proceeds with randomly initialized weights and still emits plausible-looking GIFs
and physics panels. Case 04 warns; Case 05 does not warn at all. Readers are told to
confirm `Loaded best validation model from ...` appears in the log before trusting
any output.

### Open question deliberately left to the author

Case 05's `--mode test` uses `case_06`. Case 04's test mode uses `case_07`,
`config.py` maps `"test"` to `extrapolation/case_07`, and the preprocessor builds
case 07 — which all suggests `case_name = "case_06"` at `main_dem_hard.py:175` is a
typo. It was **not** changed, because switching the evaluation case changes reported
numbers and that is the author's call. The README documents current behaviour and
points at the one-line change.

---

## 4. Data provenance

Zenodo record **19691595** (https://zenodo.org/records/19691595) — all six files
verified live (HTTP 200), 247.4 MB total.

| File | Size |
| --- | --- |
| RawData_60Spheres_Homogeneous_Interaction_Inside_Cuboidal_Enclosure.zip | 39.9 MB |
| RawData_60Spheres_Gravity_Inside_Cuboidal_Enclosure.zip | 30.3 MB |
| RawData_Extrapolation_2073Spheres_Gravity_Inside_Rotating_Cylinder.zip | 176.5 MB |
| RawData_Benchmark_1Sphere_Multiple_Wall_Collision.zip | 0.3 MB |
| RawData_Benchmark_2Spheres_Oblique_Collision.zip | 0.04 MB |
| Detailed_Information_on_Data_Structure.pdf | 0.4 MB |

Raw CSV schema confirmed against the real archives — the columns the preprocessor
reads (`coordinates:{0,1,2}`, `Velocity:{0,1,2}`, `Angular_velocity:{0,1,2}`,
`Density`) are all present. Extra columns present but unused: `Diameter`,
`Orientation:{0,1,2}`, `Particle_ID`.

Case 04 downloaded: 11,613 CSV frames
(training 7,505 / validation 1,501 / extrapolation 1,501 / benchmark 1,106).

---

## 5. Preprocessing results (real data)

Sliding (t-1, t, t+1) window drops the first and last frame of every sequence, so a
1501-frame case yields 1499 graphs.

### Case 04 — homogeneous

| Split | Graphs |
| --- | --- |
| training / case_01 … case_05 | 1499 each (7,495 total) |
| validation / case_06 | 1499 |
| extrapolation / case_07 | 1499 |
| benchmark / oblique_sphere_collisions | 99 |
| benchmark / oblique_wall_collisions / {10,30,45,60,90}_deg | 199 each (995 total) |

Note: the oblique-sphere archive contains a single `head_on_oblique_collision/`
subfolder with 101 CSVs. `get_data.py` flattens it into
`oblique_sphere_collisions/`, which is why the README's "100 files" becomes 99
graphs after the sliding window.

### Case 05 — heterogeneous gravity

| Split | Graphs |
| --- | --- |
| training / case_01 … case_05 | 1499 each |
| validation / case_06 | 1499 |
| extrapolation / case_07 | 1499 |
| rotating_cylinder | 1999 (2,073 spheres/frame) |

---

## 6. Measured cost on CPU (why training moved to a GPU server)

Measured on the Apple M4 Pro (14 cores, CPU-only), Case 04, real data:

| Quantity | Measured |
| --- | --- |
| Dataset load (7,495 train + 1,499 val graphs) | 1.5 s |
| `get_stats` over the full training set | 0.2 s |
| Model parameters | 734,996 |
| Batches per epoch (batch size 64) | 118 |
| **One training epoch** | **~65 s** |
| One validation rollout (200 autoregressive steps) | 11.9 s |

Projection for the configured 500-epoch schedule: **9.0 h training + 0.3 h
validation = ~9.3 h for Case 04 alone**, with Case 05 expected to be comparable or
slower (it adds the external-force MLP). Roughly 19-20 h for both cases on CPU.

**Decision:** training was moved to a CUDA server rather than run on CPU. The
Case 05 CPU benchmark was cancelled as no longer relevant.

### CPU vs GPU install — corrected guidance

An earlier draft of these files told GPU users to install torch via
`--index-url .../cu126`. That is **wrong as a general instruction**: the default
PyPI wheel for `linux-x86_64` *is* the CUDA build. Verified from PyPI metadata for
torch 2.7.0:

| Platform | Wheel size | Variant |
| --- | --- | --- |
| macosx_11_0_arm64 | 68.6 MB | CPU-only |
| manylinux_2_28_x86_64 | **865.2 MB** | **CUDA 12.6 bundled** |
| manylinux_2_28_aarch64 | 99.2 MB | CPU-only |
| win_amd64 | 212.5 MB | CUDA |

So on the GPU server a plain `pip install -r requirements.txt` is sufficient. The
`--index-url` override is only needed to pin a *different* CUDA version, or to force
a CPU-only build on Linux. Both spec files now say this.

---

## 7. Publishing to GitHub

Remote: https://github.com/vinay-IMOSEPFL/DynamicalGraphNet-DEM.git (was empty).

`.gitignore` excludes everything regenerable, keeping the repository at **25 files /
288 KB** instead of ~3.6 GB:

| Excluded | Size | Regenerate with |
| --- | --- | --- |
| `case_0*/data/` | ~635 MB | `get_data.py` |
| `case_0*/dataset/` | ~3.0 GB | `preprocess.py` |
| `case_0*/saved_models/` | — | `--mode train` |
| `case_0*/results/` | — | `--mode test` |

Commit authorship: the `Co-Authored-By` trailer was removed at the author's
request — commits are attributed solely to the researcher.

---

## 8. Reproduction run — status

- [x] Fresh conda env created from `environment.yml`
- [x] Env verified against `requirements.txt` pins
- [x] OpenMP conflict found and fixed
- [x] Zenodo data downloaded (both cases)
- [x] Preprocess Case 04 and Case 05 (incl. 1,999 cylinder graphs)
- [x] Measure per-epoch cost on CPU, project full training time
- [x] Push cleaned repository to GitHub
- [ ] Train Case 04 on CUDA server
- [ ] Train Case 05 on CUDA server
- [ ] Evaluate checkpoints, collect metrics, render visualizations

_(training numbers to be filled in from the server run)_

---

# Part II — CUDA server reproduction run

Everything below was measured on the GPU server, on **2026-08-06**, from a clean clone of
the published repository. Part I above was the macOS/CPU audit; this part supersedes its
"to be filled in" placeholders.

## 9. Server environment

### 9.1 Hardware and drivers

| Item | Value |
| --- | --- |
| Host | `enac-imos-gpu01` |
| GPU | 4x NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each |
| NVIDIA driver | 535.288.01 |
| Driver CUDA runtime | 12.2 |
| CPU | 16 cores |
| RAM | 62 GB |
| Disk free (before run) | 343 GB |
| OS | Linux 5.15.0-185-generic |

### 9.2 The environment file needed no changes

`conda env create -f environment.yml` succeeded as published. Verified pins: Python
3.12.13, torch **2.7.0+cu126**, torch-geometric 2.6.1, numpy 1.26.4, pandas 2.2.2,
matplotlib 3.9.2, seaborn 0.13.2, imageio 2.33.1, tqdm 4.66.5 — all matching Part I.

The Part I prediction held: the default PyPI `linux-x86_64` wheel is the CUDA build, so no
`--index-url` override was needed.

```
torch 2.7.0+cu126 | cuda available: True | device count: 4 | NVIDIA GeForce RTX 2080 Ti
```

`import torch, numpy` (the order the pipeline scripts use) works, and a real GPU matmul was
executed — `torch.cuda.is_available()` alone is not proof the stack works.
`KMP_DUPLICATE_LIB_OK` was never set.

**Mildly surprising, worth recording:** torch is built against CUDA **12.6** while the
installed driver only advertises CUDA **12.2**. This works because of CUDA minor version
compatibility — a 12.x runtime runs on any 12.x driver. No action needed, but it looks
like a mismatch at first glance.

### 9.3 Data — matches Part I exactly

Downloads completed from Zenodo record 19691595 with no deviation.

| Quantity | Expected | Measured |
| --- | --- | --- |
| Case 04 raw CSV frames | 11,613 | **11,613** |
| Case 05 raw CSV frames | 12,508 | **12,508** |
| Case 04 `data/` | — | 87 MB |
| Case 05 `data/` | — | 550 MB |

Case 04 split breakdown reproduced Part I exactly: training 7,505 / validation 1,501 /
extrapolation 1,501 / benchmark 1,106.

### 9.4 Preprocessing — matches Part I exactly

Every graph count matched the expected values, including the cylinder.

| Split | Graphs | Nodes/graph |
| --- | --- | --- |
| Case 04 training case_01…05, validation case_06, extrapolation case_07 | 1499 each | — |
| Case 04 benchmark oblique_sphere_collisions | 99 | — |
| Case 04 benchmark oblique_wall_collisions/{10,30,45,60,90}_deg | 199 each | — |
| Case 05 training case_01…05, validation case_06, extrapolation case_07 | 1499 each | 420 |
| Case 05 rotating_cylinder | **1999** | **8292** |

The cylinder's 8,292 nodes/graph confirms the documented layout: 2,073 real spheres plus
three ghost-node sets (2073 x 4 = 8292).

| Cost | Case 04 | Case 05 |
| --- | --- | --- |
| Preprocessing wall clock | 12 min 14 s | 24 min 44 s |
| Peak RSS | 1.5 GB | 3.6 GB |
| `dataset/` on disk | 373 MB | 2.6 GB |

Both preprocessors are single-threaded and CPU-bound, as Part I noted; they were run
concurrently on separate cores, which is why the two wall-clock figures overlap in time.

## 10. Deviations from the published pipeline

All four are additive. **No hyperparameter, model architecture, or evaluation protocol was
changed.** The numerical path through `evaluate_rollout` and the trainer is untouched.

1. **`main_dem_hard.py` gained an optional `--case_name` flag** (default `case_06`).
   This exists solely to let the cuboid test rollout be run against `case_07` as well —
   see §11. With the flag omitted, behaviour is bit-for-bit the shipped behaviour.

2. **Per-epoch training-loss print** added to both `main_*.py`. The training loop computed
   a loss every epoch but only printed it on validation epochs (every 5th), so 80% of the
   loss trajectory was unrecoverable from the log. This is a `print` — it cannot affect
   results.

3. **`utils/metrics_dump.py` (new)** serialises what `evaluate_rollout` already returned
   and the pipeline previously discarded: per-step scaled error sequences, their summary
   statistics, and the rollout's wall-clock cost. Reporting only; nothing feeds back.

4. **Missing-checkpoint warnings made uniform.** Case 05 printed nothing at all when the
   checkpoint was absent, and two Case 04 benchmark modes printed a non-standard string.
   All five evaluation entry points now print `Loaded best validation model from ...` on
   success and an explicit warning on failure, and the loaded/not-loaded state is recorded
   in each `metrics.json`. This closes the "silent evaluation with random weights" trap
   flagged in Part I §3.

Helper scripts added under `scripts/`: `run_training.sh` (detached launcher),
`collect_metrics.py` (aggregates every `metrics.json` into CSV/JSON/markdown),
`parse_training_log.py` (loss and validation curves).

## 11. The `case_06` question — resolved by the author

Part I flagged that `main_dem_hard.py --mode test` evaluates `case_06`, the same case used
for validation checkpoint selection, while every other signal (Case 04's test mode,
`config.py`'s `"test"` split, the preprocessor) points at `case_07`.

**Author's decision: report both, labelled separately.** The shipped `case_06` number is
kept for continuity, and `case_07` is reported as the genuinely held-out extrapolation
result. The line was therefore not "fixed" — it was made addressable via `--case_name`, so
the default path still reproduces the published behaviour exactly.

When reading the metrics table, note that **the Case 05 `case_06` row is not a
generalization result** — the checkpoint was selected on that case.

## 12. GPU vs CPU cost

Measured on Case 04, steady state, with both trainings running concurrently on separate
GPUs (host CPU shared).

| Quantity | CPU baseline (M4 Pro, 14 core) | GPU (RTX 2080 Ti) | Speedup |
| --- | --- | --- | --- |
| Model parameters | 734,996 | **734,996** (identical) | — |
| Batches/epoch (batch 64) | 118 | 118 | — |
| Dataset load | 1.5 s | 8 s | 0.2x |
| `get_stats` | 0.2 s | ~1 s | — |
| **One training epoch** | ~65 s | **~15 s** | **~4.3x** |
| **200-step validation rollout** | 11.9 s | **18.1 s** (mean of 8) | **~0.66x — slower** |

Case 05's model has **768,407** parameters; the extra ~33.4 k is the external-force
(gravity) MLP enabled by `use_ext_force=True`.

**The most interesting result here is the negative one.** The autoregressive validation
rollout is *slower* on the RTX 2080 Ti than on the M4 Pro CPU. This is not a
misconfiguration — it is inherent to the workload:

- The rollout is strictly sequential: one graph, one step at a time, no batching.
- Each of the 200 steps rebuilds the graph (`cdist` + boundary ghosts) and runs 5 message
  passing rounds over a ~420-node graph.
- At that size every kernel is tiny, so per-kernel launch latency dominates and the GPU's
  throughput advantage never materialises.

Training wins ~4.3x because batch-64 graphs are large enough to fill the device. The
practical consequence: **GPU helps training, not inference, for this model at this scale.**
Anyone reproducing only the evaluation modes should not expect a speedup, and the
2,073-sphere cylinder rollout is the one inference workload big enough to change that
conclusion (see §13).

## 13. Incident: first training run killed by session teardown

The first launch of both trainings was killed at 13:13 when the controlling process
exited — the runs were children of a shell that did not survive. Reported plainly rather
than silently restarted:

| Run | Reached | Train loss | Last validation score |
| --- | --- | --- | --- |
| Case 04 (attempt 1) | epoch 137/500 | 2.2118 -> 0.1858 | 1.2283e-01 |
| Case 05 (attempt 1) | epoch 35/500 | -> 0.1250 | 3.8460e-01 |

Neither run completed and **the pipeline has no resume capability** — checkpoints store
only `model.state_dict()`, not optimizer state or epoch index. Restarting from the partial
checkpoint would have produced a run that is neither a clean 500-epoch schedule nor
reproducible from the config, so both were **restarted from scratch**. The partial
checkpoints were deleted so a stale file could never be mistaken for a finished one, and
the attempt-1 logs were kept under `logs/interrupted/`.

Attempt 2 launched via `scripts/run_training.sh` under `setsid`, giving each run its own
session so it survives teardown of the controlling process.

## 14. Hyperparameter correction — Case 04 realigned with the paper

After the first 500-epoch run finished, the author supplied the paper's actual training
description:

> "Training was conducted for 200 epochs with a batch size of 64 and a learning rate of
> 3 x 10^-4, using the Adam optimizer." The best model was selected based on a 200-step
> rollout on the interpolation case.

`case_04_dem_simple/config.py` disagreed with this on two counts. On the author's explicit
instruction it was changed:

| Setting | Shipped | Paper / now | Note |
| --- | --- | --- | --- |
| `epochs` | 500 | **200** | changed |
| `lr` | 3e-5 | **3e-4** | changed — shipped value was 10x too small |
| `batch_size` | 64 | 64 | already correct |
| optimizer | Adam | Adam | already correct |
| model selection | 200-step rollout on validation case | same | already correct |

Nothing else was touched. This is the one intentional hyperparameter change in this
reproduction, and it was made on the author's direction, not on my own judgement.

### 14.1 The paper's hyperparameters are strictly better

Run 1 (shipped config) was **not** discarded — it is archived under
`archive_run1_lr3e-5_500ep/` with its checkpoint, full `results/` tree, training log and
reports, so the two are directly comparable.

| | Run 1 — shipped (500 ep, lr 3e-5) | Run 2 — paper (200 ep, lr 3e-4) |
| --- | --- | --- |
| Wall clock | 2 h 42 m | **1 h 09 m** |
| Train loss, final / min | 8.325e-02 / 8.070e-02 | **6.285e-02 / 5.864e-02** |
| Best validation score | 9.9374e-02 @ epoch 320 | **7.0779e-02 @ epoch 155** |
| Checkpoint saves | 7 | 9 |

A 29% better validation score in 40% of the wall clock. The shipped `lr = 3e-5` was simply
starving the run: at 500 epochs it had still not reached what the paper's setting reaches
by epoch 155.

### 14.2 Case 05 realigned as well

Case 05 was subsequently retrained on the same schedule, on the author's instruction. Its
learning rate already matched the paper, so only `epochs: 500 -> 200` changed.

Here the shorter schedule is **slightly worse**, which is the opposite of Case 04:

| | 500 epochs | 200 epochs (paper) |
| --- | --- | --- |
| Wall clock | 2 h 34 m 22 s | **1 h 05 m 37 s** |
| Train loss, final / min | 5.713e-03 / 4.640e-03 | 1.305e-02 / 1.183e-02 |
| Best validation score | 1.8283e-01 @ epoch 495 | 2.1004e-01 @ epoch 200 |

The distinction matters. Case 04 improved because its shipped `lr = 3e-5` was starving the
run — the change fixed a genuinely wrong setting. Case 05's learning rate was already
correct, so cutting 500 -> 200 epochs only removed training, and the score degraded by
about 15%. In both runs the best score fell on or near the final epoch, so Case 05 had not
converged when the schedule ended.

Reported as measured. The 200-epoch checkpoint is the one shipped, because matching the
paper's stated protocol takes precedence over a better number obtained off-protocol; the
500-epoch checkpoint is preserved under `archive_run1_lr3e-5_500ep/`.

## 15. Final training results

Both cases now run the paper's schedule: 200 epochs, lr 3e-4, batch 64, Adam.

| | Case 04 | Case 05 |
| --- | --- | --- |
| Epochs | 200 | 200 |
| Learning rate | 3e-4 | 3e-4 |
| Parameters | 734,996 | 768,407 |
| Wall clock | 1 h 09 m 29 s | 1 h 05 m 37 s |
| Exit code | 0 | 0 |
| Train loss, first -> last (min) | 1.9402 -> 6.285e-02 (5.864e-02) | 6.1848 -> 1.305e-02 (1.183e-02) |
| Best validation score | 7.0779e-02 @ epoch 155 | 2.1004e-01 @ epoch 200 |
| NaN losses | none | none |
| Tracebacks | none | none |

Curves and raw per-epoch values: `reports/case0{4,5}_training_curve.csv`,
`reports/case0{4,5}_validation_curve.csv`, `reports/case0{4,5}_training_curves.png`.

## 16. Evaluation metrics

Full table: `reports/metrics_summary.{md,csv,json}`; per-rollout detail including the
complete per-step error sequence lives in each experiment's `metrics/` folder.

All ten rollouts were confirmed to have loaded the trained checkpoint — the
`Loaded best validation model from ...` line was checked in every log, and
`checkpoint_loaded: true` is recorded in every `metrics.json`. No result below came from
randomly initialised weights.

### 16.1 What the numbers mean

`evaluate_rollout` reports **dimensionless** errors, dividing by training-set scales:

| Scale | Case 04 | Case 05 |
| --- | --- | --- |
| position (`edge_feat_max`) | 6.250e-03 m | 6.250e-03 m |
| velocity (`node_vel_max`) | 2.8137 m/s | 1.8823 m/s |
| angular velocity (`node_angvel_max`) | 700.01 rad/s | 723.10 rad/s |

The position scale is the interaction cutoff (1.25 x sphere diameter), **not** the box
size. So a scaled position MAE of 1.0 is one contact radius of error, and the 0.03 m box is
4.8 scale units across. This matters for reading the long-rollout rows honestly.

### 16.2 Results

| Case | Experiment | Steps | Wall clock (s) | Pos MAE mean / final | Vel MAE mean / final | AngVel MAE mean / final |
| --- | --- | ---: | ---: | --- | --- | --- |
| 04 | case_07 rollout (held-out) | 1499 | 138.3 | 1.766e+00 / 2.513e+00 | 1.380e-01 / 5.572e-02 | 1.583e-01 / 8.704e-02 |
| 04 | oblique sphere collisions | 99 | 9.2 | 1.010e-02 / 2.949e-02 | 6.894e-03 / 1.021e-02 | 2.224e-02 / 3.368e-02 |
| 04 | oblique wall 10° | 199 | 4.6 | 1.239e-01 / 2.789e-01 | 3.171e-02 / 3.520e-02 | 5.157e-02 / 5.670e-02 |
| 04 | oblique wall 30° | 199 | 3.8 | 5.057e-02 / 1.156e-01 | 1.326e-02 / 1.479e-02 | 1.289e-02 / 1.449e-02 |
| 04 | oblique wall 45° | 199 | 3.9 | 3.310e-02 / 7.722e-02 | 8.914e-03 / 1.013e-02 | 6.416e-03 / 7.295e-03 |
| 04 | oblique wall 60° | 199 | 4.2 | 2.687e-02 / 6.597e-02 | 7.526e-03 / 8.926e-03 | 7.016e-03 / 8.387e-03 |
| 04 | oblique wall 90° | 199 | 3.8 | 3.880e-02 / 9.103e-02 | 1.060e-02 / 1.187e-02 | 6.323e-04 / 6.056e-04 |
| 05 | cuboid case_06 **(= validation case)** | 1498 | 169.3 | 6.753e-01 / 8.795e-01 | 8.908e-02 / 4.687e-02 | 8.610e-02 / 4.935e-02 |
| 05 | rotating cylinder (extrapolation) | 1998 | **1257.0** | 2.291e+00 / 2.301e+00 | 4.101e-02 / 4.103e-02 | 3.675e-02 / 4.721e-02 |

**Reading these honestly:**

- The short benchmarks (99–199 steps) stay well under one contact radius of position error.
- The 1499-step cuboid rollouts accumulate 0.7–2.5 scale units of position error, i.e.
  roughly 4–16 mm in a 30 mm box. Over ~1500 fully autoregressive steps with no ground
  truth injected, individual particle trajectories have decorrelated. Velocity and angular
  velocity errors stay an order of magnitude smaller and *fall* toward the end of the run,
  because the gravity cases settle and late-time velocities are small.
- **The Case 05 `case_06` row is not a generalization result** (§11): the checkpoint was
  selected on that case, so it is optimistic by construction. An earlier run evaluated both
  `case_06` and `case_07` under the 500-epoch checkpoint and measured 7.416e-01 against
  1.281e+00 mean position error — the held-out case was ~1.7x worse, which is the scale of
  optimism to keep in mind when reading this row. On the author's instruction the `case_07`
  path was subsequently removed from Case 05 and only `case_06` is now reported.

### 16.3 Oblique wall impact — post-collision angular velocity

Final-step predicted vs DEM ground truth (rad/s):

| Angle | Predicted (wx, wy, wz) | DEM ground truth (wx, wy, wz) | L2 error | GT L2 norm | Rel. error |
| ---: | --- | --- | ---: | ---: | ---: |
| 10° | +2.637e+01, +1.622e+02, -3.234e+00 | 0, +2.517e+02, 0 | 9.333e+01 | 2.517e+02 | 37% |
| 30° | +6.512e+00, +2.217e+02, +2.689e+00 | 0, +2.429e+02, 0 | 2.236e+01 | 2.429e+02 | 9.2% |
| 45° | +1.383e+00, +1.997e+02, +2.303e+00 | 0, +2.113e+02, 0 | 1.194e+01 | 2.113e+02 | 5.7% |
| 60° | +5.315e+00, +1.512e+02, +2.186e+00 | 0, +1.613e+02, 0 | 1.163e+01 | 1.613e+02 | 7.2% |
| 90° | 0, 0, -1.272e+00 | 0, 0, 0 | 1.272e+00 | 0 | n/a |

Two useful physics checks:

- **90° (normal impact) induces no spin in the DEM ground truth** — all three components are
  exactly zero. The model predicts a residual |w| of 1.27 rad/s against an angular velocity
  scale of 700 rad/s, i.e. 0.18% of scale. Essentially zero, correctly.
- The ground truth spin is purely about the y axis at every angle; the model keeps the
  spurious wx/wz components small but non-zero, and its worst case is the **shallowest**
  10° impact (37% error), where the grazing contact is hardest to resolve. Accuracy is best
  in the 45–60° range.

## 17. Cost of the cylinder extrapolation

The headline inference workload, measured:

| Quantity | Value |
| --- | --- |
| Sync points | 1,998 (of 1,999 graphs; the last hits `StopIteration`) |
| Micro-steps per sync | 10 |
| **Total model forwards** | **19,980** |
| Nodes per forward | 8,292 (2,073 real + 3x ghosts) |
| Graph rebuilds (`cdist` + boundaries) | one per micro-step |
| **Wall clock** | **1,188.9 s = 19 m 49 s** |
| Rendered frames (`frequency=25`) | 80 |
| `rollout_data/` on disk | 391 MB |
| GIF | 7.4 MB |

That is ~16.8 forwards/s over an 8,292-node graph — and it is the one workload where the
GPU clearly earns its place, in contrast to the small-graph rollouts of §12.

**No render worker cap was needed.** `ProcessPoolExecutor(max_workers=os.cpu_count())`
spawned 16 matplotlib processes on this 16-core / 62 GB host; available RAM never dropped
below 52 GB during rendering. On a machine with many more cores than this, the caution in
the brief still applies.

The cylinder mode produces **frames and a GIF only, no physics panel** — matching the Part I
documentation correction, not a failure.

## 18. Artifacts produced

| Artifact | Path |
| --- | --- |
| Case 04 checkpoint (paper config) | `case_04_dem_simple/saved_models/model_checkpoint_best_val.pth` |
| Case 05 checkpoint | `case_05_dem_hard/saved_models/model_checkpoint_best_val.pth` |
| Metrics table (md / csv / json) | `reports/metrics_summary.*` |
| Training + validation curves | `reports/case0{4,5}_*_curve.csv`, `*_training_curves.png` |
| Case 04 held-out rollout GIF | `case_04_dem_simple/results/case_07_rollout/gif/trajectory_rollout_DynSolver.gif` (21 MB) |
| Case 04 physics panel | `case_04_dem_simple/results/case_07_rollout/rollout_physics_plots/` |
| Oblique sphere GIF + panel | `case_04_dem_simple/results/benchmark_oblique_sphere_collisions/` |
| Oblique wall summary plots | `case_04_dem_simple/results/benchmark_oblique_summary/` |
| Case 05 cuboid GIFs + panels | `case_05_dem_hard/results/case_0{6,7}_gravity_rollout/` |
| Cylinder GIF | `case_05_dem_hard/results/rotating_cylinder_rollout/gif/rotating_cylinder.gif` (7.4 MB) |
| Run-1 archive (superseded) | `archive_run1_lr3e-5_500ep/` |
| Training / evaluation logs | `logs/` (attempt-1 casualties under `logs/interrupted/`) |

Total `results/` footprint: 41 MB (Case 04) + 469 MB (Case 05).

## 19. On comparing these numbers to the paper

**These are a new baseline, not a validation of the paper.** The repository ships no
reference metrics and no pretrained checkpoints, so there was nothing to diff against. The
paper's *hyperparameters* were supplied by the author (§14) and have been matched for
Case 04, but its *reported error values* were not, and no claim of agreement is made here.

Anyone wanting that comparison should supply the paper's tabulated errors; the per-step
sequences needed to reproduce any aggregation are already on disk in each `metrics/`
folder.

Two caveats to carry forward:

1. **Spheres escape the enclosure during long rollouts** (§21). Any aggregate over a
   1499-step cuboid rollout includes escaped particles and should be read with that in mind.
2. **Single seed, single run.** Everything is seeded (`set_seed(100)`), but no
   seed-variance study was run, so no error bars accompany any number above.

## 21. The wall interaction range is half the particle interaction range

Investigating a visual artefact in the Case 04 rollout — spheres drifting far outside the
box and continuing to move — led to a reproducible finding about the boundary model.

### 21.1 The cutoff itself is applied correctly

Checked first, and cleanly ruled out:

| Check | Result |
| --- | --- |
| Two spheres 10 mm apart, far from all walls | 0 edges; `dv`, `dw` exactly 0; `dx - v*dt` = 0.000e+00 |
| Realised edge set vs brute force, 5 real `case_07` frames | exact match; 0 edges over the cutoff, 0 ghost-ghost |
| Ghost of sphere *i* linked to a different real sphere *j* | 0 occurrences in every frame sampled |
| Cutoff re-applied at every rollout step | yes; 1,710 isolated-node checks over 300 steps, 0 violations, worst deviation 1.5e-11 |

So `insert_sphere_sphere_edges` is correct, the ghost-ghost filter works, and the graph is
genuinely rebuilt every step. A sphere with no incident edges receives exactly zero impulse,
because `node_dv_int = m_inv * out_fij` with `out_fij` an untouched zero tensor.

### 21.2 The actual mechanism

A wall ghost is a mirror image, so for a sphere whose centre is a distance `d` from the wall
plane, the **centre-to-centre separation to its own ghost is `2d`**. That separation is then
tested against the same `THRESHOLD` used for sphere-sphere pairs. One constant therefore
encodes two different physical ranges:

| Pair | Distance measured | Edge appears when | Surface gap at onset |
| --- | --- | --- | --- |
| sphere - sphere | true centre separation | separation < 6.25 mm | **1.250 mm** |
| sphere - wall | `2d` (doubled by reflection) | `d` < 3.125 mm | **0.625 mm** |

Confirmed by direct measurement: sweeping a single sphere toward a wall, the wall edge first
appears at a 0.500 mm surface gap and is still absent at 0.625 mm; sweeping two spheres
together, they link at a 1.250 mm gap.

Nothing in the code states that walls should have half the reach of particles — it falls out
of the mirror geometry and is easy to miss.

### 21.3 Consequence: irreversible escape

At ~4 m/s and `dt` = 1e-4 s a sphere advances 0.4 mm per step, so it gets roughly 1.5 steps
inside the wall's 0.625 mm band to reverse its momentum, against ~3 steps for a particle
collision. If the impulse falls short the sphere crosses the plane; once `|d|` exceeds
3.125 mm on the far side the wall edge disappears and **no restoring force remains**.

Measured on the trained Case 04 checkpoint, rolling out `case_07`:

| Step | Spheres outside the box | Max overshoot |
| ---: | ---: | ---: |
| 49 | 1 / 60 | 0.02 mm |
| 100 | 15 / 60 | 7.83 mm |
| 300 | 15 / 60 | 53.65 mm |
| 600 | 15 / 60 | 127.35 mm |

Ground truth over the same window stays within 0.00055-0.02951 m and never leaves.

Three features identify the mechanism:

- **Irreversible.** The count reaches 15/60 by step 100 and never changes again.
- **Exactly ballistic.** Overshoot grows linearly at 23.9 mm per 100 steps = 2.39 m/s
  constant, the signature of zero net force.
- **Starts as a hairline miss.** The first crossing overshoots by 0.02 mm. It is not a
  numerical blow-up; the sphere simply clips the narrow band.

A compounding factor: training graphs are built from ground-truth positions, which never
leave the box, so the model has never seen an escaped sphere. Once one is out it is fully
out of distribution and its motion is arbitrary rather than merely inaccurate.

### 21.4 Left unchanged, pending a decision

No fix was applied. All three options change reported numbers and require re-preprocessing
**and** retraining both cases, since graph topology feeds the training set too:

1. Use a separate, doubled threshold for sphere-ghost pairs so the wall band matches the
   particle band. Smallest change that makes the two ranges physically consistent.
2. Test wall pairs against `d` rather than `2d`. Equivalent in effect, larger edit.
3. Clamp positions at the walls during rollout only. Cheapest, but masks the symptom
   instead of fixing the model.

Recommendation is (1), but whether the halved wall range was deliberate is the author's call.

## 22. Repository cleanup for publication

Done on the author's instruction, after the results above were measured:

- **`--case_name` removed.** Added earlier to run both cuboid cases for Case 05, it was
  scaffolding that would confuse a reader. `--mode test` is plain `case_06` again, and the
  `case_07` results for Case 05 were deleted.
- **Duplicate loss print removed.** Two lines were reporting the same number every fifth
  epoch. `parse_training_log.py` reads either format, so earlier logs still parse.
- **Docstrings rewritten** in both entry points, both rollout evaluators, and the Case 05
  visualization module: banner-and-narration prose replaced with concise descriptions and
  real `Args:`/`Returns:` sections.
- **Promotional wording removed** throughout ("massive", "ultra-fast", "comprehensive
  suite"); a grep for those terms now returns nothing.
- **`readme_dem.md` replaced by `README.md`**, correcting four claims that had become false:
  the hyperparameters for both cases, the "no pretrained weights ship" section, the
  case_06/case_07 note, and GPU install advice that told users to install from the cu126
  index URL — unnecessary, since the default Linux wheel is already the CUDA build (§6).
- **Checkpoints are now tracked** so both cases can be evaluated without retraining;
  `logs/` and the run archive stay ignored.

## 20. Reproduction run — final status

- [x] Fresh conda env from `environment.yml`, CUDA verified on RTX 2080 Ti
- [x] Zenodo data downloaded, frame counts matched exactly
- [x] Both cases preprocessed, all graph counts matched exactly
- [x] Case 04 trained — 200 epochs, paper hyperparameters
- [x] Case 05 trained — 200 epochs, paper hyperparameters
- [x] Case 04 evaluated: held-out case_07, oblique sphere, oblique wall x5
- [x] Case 05 evaluated: cuboid case_06, per the author's decision
- [x] Rotating cylinder extrapolation (19,980 forwards, 20 m 57 s)
- [x] Metrics collected to JSON/CSV/markdown; loss and validation curves plotted
- [x] All GIFs and physics panels rendered
- [x] Boundary interaction range investigated and documented (§21)
- [x] Repository cleaned for publication; checkpoints tracked (§22)
- [ ] Decision on the wall interaction range (§21.4) — needs re-preprocess + retrain
- [ ] Seed-variance study (none run)

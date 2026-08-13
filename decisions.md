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
- [x] Boundary-model experiments run and reverted (§23)
- [x] Reproduction scaffolding removed from the pipeline (§24)
- [ ] Decision on the wall interaction range (§21.4) — needs re-preprocess + retrain
- [ ] Seed-variance study (none run)

## 23. Boundary-model experiments — all tried, all reverted

Four modifications were tested against the shipped baseline and **none were kept**. Recorded
because the negative results are informative and cheap to repeat otherwise.

| Configuration | Best val | case_07 pos MAE (mean/final) | Escapes @600 |
| --- | --- | --- | --- |
| **Baseline (shipped, kept)** | 7.078e-02 | **1.766 / 2.513** | **15/60** |
| `edge_dx / dx_norm_max` instead of min-max scaling | 7.176e-02 | 2.093 / 3.193 | 12/60 |
| Own-image ghost edges only, 200 epochs | 8.302e-02 | 7.480 / 14.07 | 26/60 |
| Own-image + nearest wall only, 200 epochs | 1.236e-01 | 8.105 / 14.08 | 30/60 |
| Own-image only, early stopping (875 epochs) | **6.636e-02** | 5.243 / 8.994 | 18/60 |

### 23.1 The cross-image edges are load-bearing

The shipped code lets a sphere bond with the mirror image of a *different* sphere. That is
unphysical, and it is also rare in training (28 cross-image edges across all sampled
training frames) but common at rollout time — up to **172 of 216** wall edges at one step,
because predicted states let spheres overlap in ways ground-truth DEM never does.

Removing them alone made rollouts **worse**, not better. A controlled isolation test —
same checkpoint, only the rollout topology changed — attributed the bulk of the damage to
the topology itself rather than to run-to-run variance:

| Configuration | case_07 pos MAE mean |
| --- | --- |
| baseline checkpoint + baseline topology | 1.766 |
| baseline checkpoint + own-image topology | **5.869** (3.3x worse) |
| own-image checkpoint + own-image topology | 7.480 |

Interpretation: the spurious edges were supplying extra wall repulsion that partially
compensates for how briefly a sphere stays inside the interaction band. Removing them
exposes the under-resolved contact instead of fixing it.

### 23.2 Longer training helps but does not close the gap

Given 10,000 epochs with early stopping (patience 50 evaluations = 250 epochs), the
own-image model stopped at epoch 875 with its best score at epoch 625. It reached the
**best validation score of the whole study** (6.636e-02, better than baseline's 7.078e-02)
yet remained **3.0x worse** on the 1499-step `case_07` rollout.

That split is the useful finding: validation is a 200-step rollout, the test is 1499 steps.
The physically-cleaner graph yields a better short-horizon model and a less stable
long-horizon one. Both statements are true; they measure different things.

### 23.3 Correction to §21

§21 called the doubled sphere-ghost separation a defect that "halves the wall band". That
framing was wrong. A sphere touching the wall sits at `d = r`, so its image is `2d = 5.0e-3`
away — exactly the centre distance of two spheres in contact. In image space the wall is an
identical mirror sphere, with contact and cutoff at the same values as a particle pair. The
image method is correct; what is genuinely short is the **time** spent in the band
(~1.6 timesteps at 4 m/s), and a sphere-sphere collision at the same relative speed gets the
same 1.6 steps. The asymmetry claimed in §21 does not exist.

**Decision: revert to the shipped baseline.** It has the best long-horizon rollout, and the
evidence shows the escape behaviour is a contact-resolution problem rather than something
the ghost-edge topology can fix on its own.

## 24. Reproduction scaffolding removed

The instrumentation added for this reproduction was stripped so the published pipeline is
the author's original code:

- `utils/metrics_dump.py`, `scripts/` and `reports/` deleted.
- `main_dem_simple.py` and `main_dem_hard.py` restored to their pre-reproduction state —
  no timing wrappers, no metric dumps, no per-epoch loss print, no `checkpoint_loaded` flag.

Retained, because they are not scaffolding: the three blocking-defect fixes (§1), the
paper-aligned hyperparameters (§14), the docstring cleanup (§22), the trained checkpoints,
`README.md` and this log.

One consequence to be aware of: restoring the original checkpoint-loading blocks reinstates
the silent-evaluation behaviour of §3 — Case 05 prints **nothing** when the checkpoint is
missing and proceeds with random weights. This is the author's original code and is
documented in `README.md`, but it is now shipped deliberately rather than fixed.

## 25. The 500-epoch run, and the full rollout suite on its checkpoint

Run folder `12_500ep`. Identical to `11_paper_200ep` in every respect except the epoch
ceiling (200 -> 500): same corrected `model_dem.py` (602,384 params), same mirror-reflection
boundary with own-image edges only, same dataset build, lr 3e-4, batch 64, nf 128, 5 message
passes.

Stopped by hand at ~epoch 315 of 500, best validation score **1.0096e-01**. It was not run to
the schedule's end, so this is a partial run and is labelled as such. Best-val history: flat at
1.0683e-01 from ~epoch 225 to ~epoch 295, then improved again to 1.0096e-01 by ~epoch 315 --
the curve had not converged when the run was stopped.

### 25.1 Longer training did not help

600-step diagnostic, `12_500ep` against `11_paper_200ep`:

| run | epochs | best val | posMAE (600) | escapes | KE ratio (600) |
|---|---|---|---|---|---|
| 11_paper_200ep | 200 (complete)  | 9.849e-02 | 2.538  | 15/60 | 17.94x |
| 12_500ep       | ~315 (stopped)  | 1.0096e-01| 2.5367 | 16/60 | 16.62x |

The three rollout metrics are indistinguishable. Both runs sit inside the run-to-run spread
produced by CUDA `index_add_` atomics, so this pair does not separate "500 epochs helps" from
noise -- it only shows that the extra 115 epochs bought nothing measurable. Note the best
validation score is slightly *worse* at 315 epochs than at 200, on a metric evaluated by
200-step rollout; the two are separate rollouts of a non-deterministic kernel.

### 25.2 Full rollout suite

All four Case 04 rollouts were run on the `12_500ep` checkpoint, plus the escape/energy
diagnostic. Every mode was confirmed to load the checkpoint (the two benchmark modes print
"Loaded best model", without the word "validation" -- an inconsistency in the log strings, not
a missing load). Zero "Checkpoint not found" warnings, all five stages exit 0.

`main_dem_simple.py` has no mode for case_06, the case used for checkpoint selection, so
`rollout_case06.py` was added. It reuses `evaluate_rollout`, `set_seed(100)`, and the boundary
setup from the shipped test mode unchanged.

Neither `evaluate_rollout` nor `main_dem_simple.py` ever prints or persists the mean rollout
error; the per-step MAEs survive only inside `rollout_data/snapshot_*.pt`. The figures below
are aggregated from those snapshots.

| rollout | steps | scaled MAE pos | vel | angvel |
|---|---|---|---|---|
| case_07 (held out, extrapolation) | 1499 | **5.6092** | 0.2775 | 0.1891 |
| case_06 (validation, selected on) | 1499 | 0.7867 | 0.0695 | 0.0716 |

case_06 is the case the checkpoint was selected on. Its errors are optimistic by construction
and are not a generalization result. The 7x gap to case_07 is consistent with case_07 living
under `homogeneous/extrapolation/`.

### 25.3 The honest comparison against the shipped baseline

Earlier entries compared 600-step diagnostics from modified runs against the baseline's
1499-step number, which is not a like-for-like comparison. With the 1499-step case_07 figure
now in hand for a modified run:

| | case_07 posMAE @ 1499 |
|---|---|
| shipped baseline (commits 565385a, 76ed5f9) | 1.766 |
| 12_500ep (corrected model + modified boundary) | 5.6092 |

The corrected-model / modified-boundary pipeline is roughly **3.2x worse** than the shipped
baseline on the held-out case at the full evaluation horizon. This is the clearest statement of
the regression available so far, and it is not favourable to the modifications.

Two caveats, both of which cut against over-reading the number. First, the baseline 1.766 was
produced by the committed code -- original model *and* original boundary -- so this comparison
confounds the model correction with the boundary change; it does not attribute the regression
to either one. Second, `12_500ep` is a partial run stopped mid-improvement. The corresponding
1499-step figure for the completed `11_paper_200ep` run cannot be recovered without re-running
it, because this suite overwrote `results/case_07_rollout/rollout_data/`; its checkpoint is
preserved at `11_paper_200ep/model_checkpoint_best_val.pth` if that number is wanted.

Consistency check: the diagnostic and the evaluator ran case_07 independently and agree to
within atomics noise over the same window -- 2.5367 vs 2.6480 mean posMAE over 600 steps.

### 25.4 Escape and energy behaviour is unchanged

```
step  outside   overshoot    KE pred     KE gt    ratio    posMAE
100     16/60      32.6mm     493.51    460.63    1.07x    0.201
600     16/60     370.0mm     320.81     19.30   16.62x    2.537
```

The escape count is fixed at 16/60 from step 100 onward -- the spheres that leave never return,
and their overshoot grows linearly (ballistic drift at constant velocity). Ground truth sheds
96% of its kinetic energy over 600 steps (460 -> 19); the model sheds 35% (494 -> 321). The
missing dissipation, documented in §23, is untouched by the longer schedule.

## 26. Why the particles escape: the ghost wall is a one-way trapdoor

§21 established that the interaction cutoff is applied correctly and §23 that escapes track
missing dissipation. Neither identified the actual mechanism. This section does, by probing the
learned wall force directly rather than inferring it from rollouts.

### 26.1 The wall force is an odd function of the signed distance

A single sphere was placed at a sweep of positions across the +x wall (plane at x = 30mm,
radius 2.5mm, so first contact at x = 27.5mm) and the predicted `dv_x` read off.
Checkpoint `12_500ep`, sphere at rest:

```
 x (mm)   d=x-30   edges         dv_x     verdict
   26.8    -3.20       0    0.000e+00     no edge - free flight
   27.5    -2.50       2   -1.960e-03     restoring (inward)
   28.5    -1.50       2   -3.780e-01     restoring (inward)   <- peak
   29.5    -0.50       2   -1.954e-01     restoring (inward)
   29.9    -0.10       2   -3.308e-02     restoring (inward)
   30.1    +0.10       2   +3.308e-02     EXPELLING (outward)
   30.5    +0.50       2   +1.954e-01     EXPELLING (outward)
   31.5    +1.50       2   +3.780e-01     EXPELLING (outward)
   33.2    +3.20       0    0.000e+00     no edge - free flight
```

The force is exactly antisymmetric about the wall plane: `dv_x(+d) = -dv_x(-d)` to every digit
printed. This is structural, not a coincidence of training. The configuration at `+d` is the
mirror image of the configuration at `-d` -- the sphere and its ghost simply swap sides -- and
the network is reflection-equivariant, so its output is forced to mirror with it. Nothing in
the graph records which side of the plane the sphere is on: the ghost is always at `p - 2d n`,
and `node_feat` marks a node as ghost but not the *sign* of the offset.

Three consequences follow immediately, and they compound:

1. **The restoring force vanishes exactly at the wall plane.** An odd function is zero at the
   origin. The barrier is weakest precisely where it needs to be strongest.
2. **Past the plane the force reverses.** For `d > 0` the ghost lies *inside* the box, so the
   repulsion that had been pushing the sphere inward now pushes it outward. The wall does not
   merely fail to stop the sphere; it actively ejects it.
3. **Past `d = +3.125mm` the edge disappears.** The sphere-to-ghost separation is `2d`, so the
   `THRESHOLD` of 6.25mm is exceeded at `d = 3.125mm` and no wall edge can ever form again.
   Free flight, permanently.

Together: a finite barrier that vanishes at its peak of need, then an active ejector, then a
point of no return. Once a sphere crosses the plane it is gone.

### 26.2 The pass-through speed is below the speeds in the data

A single sphere fired head-on at the wall bounces or punches through depending only on speed.
Bisected threshold: **3.339 m/s**. Below it the rebound is stable with coefficient of
restitution 0.79-0.87; above it the sphere exits and drifts forever.

The reason is an impulse budget. The contact band from first touch to the wall plane is 2.5mm,
covered in `2.5e-3 / (v * 1e-4) = 25/v` steps. The mean restoring `|dv|` over that band is
0.231 m/s per step. A rebound needs a total `dv` of `2v`. The available impulse falls as `1/v`
while the requirement grows as `v`, so above a critical speed the budget cannot be met.

**Every sphere in case_07 starts at 3.811 m/s** -- above the 3.339 m/s threshold. The entire
population begins the simulation fast enough to punch through any wall it meets head-on.

### 26.3 The escaped set is exactly the fast set

From the saved case_07 rollout:

```
step  outside/60   fast(>3.34 m/s)   outside&fast   inside&fast   mean|v| out   mean|v| in
 100     17            17                17              0           4.597        1.762
 300     17            17                17              0           4.473        0.900
 600     17            15                15              0           4.279        0.522
```

`inside & fast` is zero at every step. Every sphere still in the box is below the pass-through
speed; every sphere above it has already left. The escape count freezes after ~step 100 not
because the boundary starts working, but because the population that could escape already has.
The escapees keep their speed almost exactly (4.60 -> 4.28 over 500 steps, the residue of
leaving the band), which is the ballistic drift seen in the trajectories, while the survivors
decay from 1.76 to 0.52.

This also explains why the escape count never recovers and why all eight boundary
reformulations in §23 failed to change it: none of them removed the sign flip.

### 26.4 This is pre-existing, not introduced by the modifications

The same probe was run against the shipped baseline -- HEAD's `boundary_model.py` (cdist edges
with the ghost-ghost filter, frozen zero ghost kinematics), HEAD's `model_dem.py`, and the
Aug 6 checkpoint, which predates every boundary change made in this reproduction:

```
   29.9    -0.10       2   -2.840e-02     restoring (inward)
   30.1    +0.10       2   +2.840e-02     EXPELLING (outward)
   31.5    +1.50       2   +3.929e-01     EXPELLING (outward)
```

Identical antisymmetry, identical sign flip, identical loss of edge beyond 3.125mm. The
baseline's 15/60 escapes arise from the same mechanism. **This is a property of the ghost-node
boundary as published**, and it is why the reflection formula, the edge-pairing rule, and the
ghost kinematics could all be varied in §23 without moving the escape count: every variant
kept the mirror construction, and the mirror construction is what forces the force to be odd.

### 26.5 What would fix it -- not applied

No hyperparameter, architecture, or protocol change has been made; the following is reported
for a decision, per the standing instruction.

- **Break the mirror degeneracy.** Give the interaction the sign of `d` -- as an edge feature,
  or by distinguishing the ghost's node type by side. This is the direct fix: it removes the
  constraint that forces the force to be odd, letting the network learn a barrier that stays
  repulsive across the plane. It changes the input features, so it requires retraining.
- **Raise the impulse budget.** `v_crit` scales roughly as `sqrt(k)` in the number of
  integration sub-steps, so `k = 2` moves it to about 4.7 m/s, above the 3.811 m/s initial
  speed. Case 05 already runs 10 micro-steps. In this codebase `num_msgs` sets both the
  sub-step count and the message-passing depth, so changing it invalidates the checkpoint.
- **Project positions back into the box** after each step. This suppresses the symptom without
  addressing the cause and would corrupt the momentum bookkeeping; recorded only for
  completeness.

The first is the real fix. Note that it is a change to the published method, not a bug fix in
its implementation -- the implementation faithfully realises the method described.

## 27. Correction to §23, and the two changes worth making

### 27.1 The "missing dissipation" finding in §23 was an artifact

§23 reported 13-42x excess kinetic energy across every run and concluded that dissipation, not
the boundary, was the gap. That conclusion was wrong. The KE sum was taken over all 60 spheres,
including the ones that had already escaped and were drifting ballistically at 4+ m/s. Splitting
the case_07 rollout by escaped vs still-inside:

```
step  esc |  KE pred   KE gt   ratio |  KE inside  ratio_in | %KE from esc | MAE all  MAE in
 100   17 |   523.30  462.06   1.13x |    153.05     0.43x |       70.8%  |   0.892   0.579
 300   17 |   394.12   80.85   4.87x |     43.00     0.71x |       89.1%  |   2.773   1.086
 600   17 |   335.70   19.15  17.53x |     13.54     1.05x |       96.0%  |   4.856   1.328
1200   17 |   277.50    4.58  60.63x |      3.83     1.22x |       98.6%  |   8.491   1.356
1498   17 |   257.57    2.68  96.16x |      2.35     1.15x |       99.1%  |  10.207   1.373
```

For the spheres that stay inside the box the energy ratio is **1.0-1.2x from step 400 onward**.
The model dissipates energy correctly. By step 1498, 99.1% of the apparent KE excess is carried
by the 17 escapees.

The same holds for accuracy. Position MAE over the whole system grows without bound
(0.89 -> 10.21), but restricted to the spheres still inside it **plateaus**: 1.328 at step 600,
1.356 at 1200, 1.373 at 1498. The long-horizon error growth is not error accumulation in the
learned dynamics; it is 17 spheres flying away in straight lines.

This retracts the §23 conclusion and the associated recommendation to pursue the training
objective (noise injection / multi-step rollout loss). That lever was inferred from a statistic
that double-counted the boundary failure. The boundary is the gap.

### 27.2 Smallest change: one line, no retraining

The ghost is placed at `p - 2*d*n_hat` with `d` signed, so it crosses to the inside of the box
as soon as the sphere crosses the plane -- §26's sign flip. Placing it at `p + 2*|d|*n_hat`
keeps it on the far side of the wall always:

```python
p_refl = (p + 2 * coeff.abs() * n_unit.unsqueeze(1)).reshape(-1, 3)
```

For `d < 0` the two expressions are algebraically identical. Every state seen during training
has `d < 0`, so the training distribution is untouched and **the existing checkpoint remains
valid** -- this is an inference-time change. They differ only after a sphere has crossed the
plane, where the current form expels and this one restores.

Measured on the `12_500ep` checkpoint, no retraining, 600-step case_07 rollout:

| | current | one-sided ghost |
|---|---|---|
| force at d = +1.0mm | +3.701e-01 (expelling) | -3.701e-01 (restoring) |
| head-on pass-through speed | 3.34 m/s | **4.75 m/s** |
| escaped at step 600 | 16/60 | **6/60** |
| KE ratio at step 600 | 16.62x | **3.43x** |
| scaled posMAE over 600 steps | 2.537 | **1.304** |

The pass-through speed moves above the 3.811 m/s initial speed of the data, which is why most
escapes disappear. Position error roughly halves, and the whole-system MAE (1.304) now sits at
the level previously reached only by excluding escapees (1.328). One sphere that had already
crossed was pulled back in between steps 100 and 300 -- the boundary is no longer a trapdoor.

### 27.3 Biggest change: retrain against a boundary that cannot expel

The one-line fix is bounded by two things it does not address. The barrier still vanishes
exactly at the plane, and it still disappears beyond `|d| = 3.125mm`, so a fast enough sphere
still leaves -- hence the residual 6/60. And the model has never seen `d > 0` in training, so
its behaviour there is only as good as the mirror symmetry makes it.

The larger change is to encode which side of the wall the sphere is on -- the sign of `d` as an
edge feature, or a distinct node type for a ghost whose parent has crossed -- and retrain. That
removes the constraint forcing the force to be odd, letting the barrier stay repulsive and
non-zero across the plane instead of passing through zero at it. Expected to take the residual
escapes to zero and the system MAE to the plateau value near 1.3, with the KE ratio near 1.

Neither change has been applied. §27.2 was measured in a scratch subclass; no repository file
was modified, and no hyperparameter, architecture, or protocol was changed.

## 28. One-sided ghost applied -- results

The change of §27.2 was applied to `SphereWallInteraction.reflect` in both `case_04_dem_simple`
and `case_05_dem_hard`:

```python
p_refl = (p + 2 * coeff.abs() * n_unit.unsqueeze(1)).reshape(-1, 3)
```

`CylinderInteraction.reflect` in case_05 is a different geometry and was left untouched.

No retraining. Run folder `13_onesided_ghost` uses the `12_500ep` weights byte-for-byte; the
boundary is the only difference. The dataset was not rebuilt, and does not need to be: across
all 8994 stored training and validation graphs the worst centre excursion is -1.31 mm, so no
stored state has d > 0 and the signed and absolute forms are identical on every graph on disk.

All four rollouts plus the diagnostic completed, exit 0, four checkpoint loads, no tracebacks.

### 28.1 Results

| metric | 12_500ep (signed) | 13_onesided_ghost | shipped baseline |
|---|---|---|---|
| case_07 posMAE @ 1499 | 5.6092 | **1.7278** | 1.766 |
| case_07 vel / angvel MAE @ 1499 | 0.2775 / 0.1891 | **0.1477 / 0.1509** | -- |
| case_06 posMAE @ 1499 | 0.7867 | 0.8503 | -- |
| escaped @ step 1498 | 17/60 | **3/60** | 15/60 |
| KE ratio @ step 1498 | 96.16x | **7.86x** | -- |
| posMAE over 600 steps | 2.537 | **1.304** | -- |
| escaped @ step 600 | 16/60 | **6/60** | -- |
| KE ratio @ step 600 | 16.62x | **3.43x** | -- |

Held-out case_07 error at the full evaluation horizon falls by 69%, from 5.6092 to 1.7278.
That brings the corrected model to 1.7278 against the shipped baseline's 1.766 on the same
1499-step protocol -- from 3.2x worse (§25.3) to level. Escapes fall from 17 to 3, and velocity
and angular-velocity errors roughly halve.

case_06 is marginally *worse*, 0.7867 -> 0.8503. It is the validation case the checkpoint was
selected on and it produced few escapes either way, so the boundary change has little to fix
there; the difference is small and of the order of the run-to-run spread from atomics
non-determinism. Reported rather than set aside.

### 28.2 What the residual error is

Splitting case_07 by escaped vs still-inside confirms the §27.1 picture holds after the fix:

```
step  esc |  KE pred   KE gt  ratio | KE inside ratio_in | %KE from esc | MAE all  MAE in
 300    3 |   108.40   80.85  1.34x |     79.45   1.01x |       26.7%  |   1.237   1.035
 600    3 |    37.31   19.15  1.95x |     13.69   0.75x |       63.3%  |   1.859   1.414
1498    3 |    21.06    2.68  7.86x |      2.08   0.83x |       90.1%  |   2.421   1.441
```

The inside-sphere energy ratio stays near 1 (0.62-1.05 throughout) and the inside-sphere
position error plateaus at 1.44 (1.414 at step 600, 1.437 at 900, 1.441 at 1498). 90% of the
remaining KE excess is still carried by the 3 spheres that escape -- the same mechanism, now
affecting a sixth as many particles. Closing that last gap is the §27.3 change: encode the side
of the wall and retrain, so the barrier no longer has to vanish at the plane.

## 29. Audit after reverting r0_ij to the paper's learned weights

The analytic inverse-mass centre of §4 was reverted to the paper's learned weighted centre:
`r0_ij = (w_s x_s + w_r x_r) / (w_s + w_r)`, with `w` decoded from the static node latent.
`lambda_ij` stays removed. Parameter count 602,384 -> 668,690.

### 29.1 Blocking defect, fixed

The revert reordered `InteractionDecoder.forward` to take `(edge_index, node_latent, ...)`
but the call site in `InteractionBlock.forward` was not updated, so `edge_index` was never
passed:

```
TypeError: InteractionDecoder.forward() missing 1 required positional argument: 'interaction_latent'
```

The model would not run at all -- not a subtle numerical issue, an immediate crash on any
forward pass. Fixed by passing `edge_index` at the call site. A now-unused
`senders, receivers = edge_index` left in `InteractionBlock.forward` was removed.

### 29.2 Verification after the fix

Random weights, case_07 graph, 420 nodes:

| check | result |
|---|---|
| forward pass | OK, all finite |
| linear momentum, sum abs dP | 1.311e-06 |
| angular momentum, sum abs dL | 9.388e-07 |
| frame orthonormality | max dev 1.19e-07; a.b, a.c, b.c all < 7.5e-08 |
| frame antisymmetry under endpoint swap | exactly 0 for a, b and c |
| interaction latent exchange-invariance | exactly 0 |

Conservation is intact with the learned weights. It does not depend on the particular choice
of `r0_ij` -- only on `r0_ij` being symmetric under sender/receiver exchange, which it is,
since one shared MLP produces both weights and the expression is symmetric. What conservation
does depend on is the cross-term coefficient being exactly one, which is why `lambda_ij`
stays out.

### 29.3 Two properties worth knowing

**The learned weights collapse to two values.** `node_latent` is a function of node type
alone, so `w` takes exactly two values across the whole graph (0.2320 and 0.2383 at
initialisation). Sphere-sphere pairs therefore sit on the exact midpoint -- measured
`|r0 - midpoint| = 2.6e-09 m` against a 0.03 m box -- and sphere-ghost pairs on one fixed
point along the edge. The weighted centre has far less freedom than it appears to.

The denominator `w_s + w_r` is unconstrained in sign. At initialisation it is healthy
(min 0.464, no sign changes), but nothing prevents it from approaching zero during training
and throwing `r0_ij` far from the contact. Conservation would survive that; conditioning
would not. Not changed -- it is the published formulation.

**The edge frame is orthonormal but not consistently oriented.** Working through the
construction, `vector_c = sign(b . a) * n(a x vector_b)`, so `det(a, b, c) = sign(b . a)`.
Sampled over 2608 edges: 49.7% right-handed, 49.7% left-handed. The orientation flips
whenever `b . a` crosses zero, which it does both across edges and along a trajectory, and
the decoded impulse is generically discontinuous there.

This is not an invariance bug -- `b . a` is invariant under proper rotations, so SE(3)
invariance holds, as the earlier invariance test confirmed. Nor is the sign factor removable:
building `c` as `n(a x vector_b)` would make it symmetric under endpoint exchange rather than
antisymmetric, breaking the frame's antisymmetry and with it conservation. It is an inherent
property of this frame construction, recorded for the record rather than as something to
patch.

## 30. Integrating the boundary ghosts: tested, understood, reverted

### 30.1 Correction to an earlier measurement

Section notes above reported that integrating the ghosts made each image track its parent's
mirror about 5x more closely (8.44e-05 m of deviation against 4.49e-04 m for a frozen ghost).
That measurement was taken on a randomly initialised network and does not survive contact with
a trained one. It should not have been used to motivate the change.

### 30.2 What integration actually does

A single sphere was placed at x = 28.5 mm, moving at 1 m/s toward the wall at 30 mm, and one
model step was traced sub-step by sub-step on trained weights.

```
                         frozen                          integrated
 sub   x_sphere   x_ghost   v_ghost      x_sphere   x_ghost   v_ghost
   1   28.51570   31.50000   -1.0000     28.51570   31.52794    3.7941
   3   28.53835   31.50000   -1.0000     28.53838   33.50585   80.7875
   5   28.56072   31.50000   -1.0000     28.56094   36.92574   87.9516
```

The integrated ghost is not an image; it is a free particle being shoved away by the contact
impulse. Within one step it accelerates from -1 m/s to 88 m/s and travels 5.4 mm. Retraining
under integration tames but does not fix it -- the ghost reaches 3.7 m/s instead of 88, and the
gap still opens rather than closes.

The cause is the decoded inverse mass. Ghost nodes get their own value from the node latent,
and it is far larger than the spheres':

```
                      layer                inv_mass sphere   ghost    ratio
frozen-trained   interaction_init_layer          0.0506     0.5643      11.2
frozen-trained   interaction_proc_layer          0.0000     0.5755   12832.2
integ-trained    interaction_init_layer          0.0375     0.3158       8.4
integ-trained    interaction_proc_layer          0.0001     0.0034      67.5
```

Under the frozen scheme the ghost's inverse mass is never used, so training never constrains
it. Switching integration on promotes an unconstrained output to a load-bearing one.

### 30.3 The correct treatment, and why it does not matter

A ghost is its parent's mirror, so its displacement should be the mirror of its parent's. Then
the sphere-image gap closes at twice the rate the sphere advances, matching the two-body
problem the image method is built on. Measuring that ratio over one step:

| treatment | gap closure / sphere advance | correct value |
|---|---|---|
| frozen | 1.000 | 2.0 |
| integrated | -3 to -125 (gap opens) | 2.0 |
| mirrored displacement | 2.000 | 2.0 |

Frozen closes the contact at half the true rate; integration opens it. Only mirroring is right.

But the impulse the sphere receives is the same in all three:

```
  x(mm)   d(mm) |  dv FROZEN   dv INTEG  dv MIRROR
   28.5   -1.50 |    -0.4436    -0.4362    -0.4437
   29.0   -1.00 |    -0.4320    -0.4296    -0.4319
   29.5   -0.50 |    -0.2201    -0.2200    -0.2198
```

Frozen and mirrored agree to four decimals; integration differs by at most 1.7%. The reason is
scale: over one step of 1e-4 s the sphere advances about 60 um while the sphere-ghost gap is
about 3000 um, so intra-step ghost motion perturbs the geometry by roughly 2% at worst. The
graph is rebuilt every step, which re-derives every ghost exactly, so the error never
accumulates.

That is the explanation for the inconclusive A/B in the previous section. The treatment being
varied has around 2% leverage on the only quantity that propagates, and the measurement spread
between two rollouts of one checkpoint was 26.5%.

### 30.4 Reverted

`model_dem.py` is back to holding ghosts fixed, and is now identical to the frozen control tree
(verified by comment-stripped diff). The frozen arm's checkpoint loads and runs against it, and
ghost rows are zero again.

Mirroring the ghost displacement is the physically correct option and is cheap to implement --
the boundary model already knows each ghost's parent and wall normal, they would just need to
be carried on the graph. It is not worth doing for accuracy, since it lands within four decimals
of the frozen behaviour. It would be worth doing for correctness if this part of the model is
ever described in print.

## 31. Final shipping run (16_final_250ep)

Both cases trained from scratch at 250 epochs, lr 3e-4, batch 64, nf 128, num_msgs 5, on the
cleaned tree with the one-sided ghost boundary and ghosts held fixed during integration.
case_04 has 668,690 parameters, case_05 702,101 (the extra external-force head).

Launched with `epochs=500` and stopped once epoch 250 was logged. The stop is polled, so
training overran slightly, to epoch 255 for case_04 and 260 for case_05. This does not affect
the shipped weights: checkpoints are written only on validation improvement, and the last
improvement was epoch 210 for case_04 and epoch 55 for case_05. The shipped checkpoints are
what a 250-epoch run produces, matching the `epochs: 250` now in both configs and the README.

Incident: the run folder was renamed from `16_final_500ep` to `16_final_250ep` while both
`run.sh` processes were live. Each had already bound `OUT=16_final_500ep/...`, so every
evaluation stage failed on a missing directory. Training was unaffected and the checkpoints
were written to `saved_models/` as usual; evaluation was relaunched from a separate `eval.sh`
against those checkpoints. Every mode confirmed `Loaded best validation model`, no
`Checkpoint not found` warnings, all stages exit 0.

### 31.1 Case 04

| metric | 16_final (250 ep) | 15_frozen (200 ep) | shipped baseline |
|---|---|---|---|
| best validation | **9.3425e-02** | 9.7478e-02 | -- |
| case_07 posMAE @1499 | 2.0087 | **1.6148** | 1.766 |
| vel / angvel @1499 | 0.1597 / 0.1695 | 0.1422 / 0.1575 | -- |
| escaped @1498 | 6/60 | **4/60** | 15/60 |
| inside-sphere MAE @1498 | 1.469 | 1.354 | -- |
| posMAE over 600 steps | **1.2589** | 1.4242 | -- |

The better validation score did not carry over to the held-out rollout. The 250-epoch run wins
on the metric used to select it and on the 600-step diagnostic, and loses on the 1499-step
case_07 figure. Two things argue against reading much into that: the gap is 24%, against a
26.5% spread measured between two rollouts of a single checkpoint (§30.3), and it is driven
almost entirely by six escapees rather than four, a discrete event that dominates late-horizon
error. Single seed each, so the epoch count is not established as the cause.

Energy behaves as §27 describes: inside-sphere KE ratio 0.94x at step 600 and 0.68x at 1498,
inside-sphere MAE plateauing near 1.47, and 93.9% of the apparent late-time energy excess
carried by the six escaped spheres.

### 31.2 Case 05

Note on units: `case_05_dem_hard/rollout_evaluator.py` persists *absolute* errors in its
snapshots (`mae_pos` in mm, `mae_vel` in m/s, `mae_ang` in rad/s) and keeps the dimensionless
ones only in the returned lists, which are never written. Reading the stored values as
dimensionless overstates the angular error by the scale factor, 723.1 rad/s. Both forms below.

| rollout | steps | absolute pos | scaled pos | scaled vel | scaled angvel |
|---|---|---|---|---|---|
| case_06 cuboid, gravity | 1498 | 6.03 mm | 0.9642 | 0.1157 | 0.1162 |
| rotating cylinder, zero-shot | 1998 | 151.3 mm | 24.2118 | 0.1633 | 0.0910 |

case_06 is the case the checkpoint is selected on, so 0.9642 is optimistic and is not a
generalization result. The cylinder is a genuine zero-shot extrapolation to 2,073 spheres over
20,000 forward passes; its error grows from 2.06 at 100 steps to 24.21 averaged over the full
1998, reaching 71.59 at the final step.

### 31.3 Open concern: case_05 stops improving at epoch 55

case_05's validation score reached 3.9410e-01 at epoch 55 and never improved over the
remaining 195 epochs, while training loss kept falling (1.31e-01 to 1.18e-01 across the last
35 epochs). That pattern indicates overfitting after epoch 55 rather than convergence, and it
means the 250-epoch schedule does nothing for case_05 beyond the first fifth. A shorter
schedule or early stopping would produce the same checkpoint for a fraction of the compute.
Not changed, since the schedule is now stated in the config and the README.

## 32. Case 04 shipping run with stationary ghosts (17_ghost_zeros)

Ghost kinematics reverted to zero, one-sided reflection retained, dataset rebuilt, 250 epochs.
Best validation 7.3933e-02 at epoch 250.

### 32.1 The wall benchmark, which is what motivated the change

Y-component angular velocity after impact, rad/s:

| angle | 10 | 30 | 45 | 60 | 90 |
|---|---|---|---|---|---|
| DEM ground truth | 252 | 243 | 211 | 162 | 0 |
| stationary ghosts (this run) | ~191 | ~228 | ~201 | ~150 | 0 |
| Aug 6 original run | ~168 | ~205 | ~196 | ~141 | 0 |
| mirrored ghosts (defect) | ~65 | ~85 | ~59 | ~0 | 0 |

The response tracks ground truth from 30 degrees on and exceeds the original run at every
angle. The 10 degree point remains low, 191 against 252; that is the grazing impact with the
shortest contact.

The spurious X and Z components, which should be identically zero, also shrank: X now spans
-1.8 to +9.7 against -35 to +10 with mirrored ghosts and 0 to +14.7 in the original run; Z
spans -1.4 to +7.8 against -0.4 to +6.8 originally. These remain a genuine limitation of the
model, present in every run including the original, and are not addressed by this change.

### 32.2 Rollout metrics

| metric | 17 zero ghosts | 16_final mirrored | 15_frozen mirrored | shipped baseline |
|---|---|---|---|---|
| epochs | 250 | 250 | 200 | -- |
| best validation | **7.3933e-02** | 9.3425e-02 | 9.7478e-02 | -- |
| case_07 posMAE @1499 | **1.6149** | 2.0087 | 1.6148 | 1.766 |
| vel / angvel @1499 | **0.1374 / 0.1605** | 0.1597 / 0.1695 | 0.1422 / 0.1575 | -- |
| posMAE over 600 steps | **1.0369** | 1.2589 | 1.4242 | -- |
| escaped @1498 | **4/60** | 6/60 | 4/60 | 15/60 |
| KE ratio @600 | **1.29x** | 2.74x | 3.63x | -- |
| KE ratio @1498 | **3.77x** | 10.30x | 4.07x | -- |
| inside-sphere MAE @1498 | **1.341** | 1.469 | 1.354 | -- |

Best on every metric. The energy trace is the clearest improvement: the ratio over all spheres
stays within 1.03x to 1.29x through step 600, where the mirrored-ghost run reached 2.74x, and
the inside-sphere ratio holds at 0.57 to 0.61 late in the rollout.

The 1499-step figure, 1.6149, is indistinguishable from 15_frozen's 1.6148. That metric is
dominated by however many spheres escape, four in both cases, so it does not separate the two.
Everything that is not escape-dominated -- validation, the 600-step mean, energy, and the wall
benchmark -- favours stationary ghosts.

### 32.3 Dataset provenance

case_04's graphs were rebuilt at 12:42-12:46 on 2026-08-13, and every split was checked: no
stale files, ghost kinematics zero throughout, graph counts correct, and stored graphs
reproducing the live boundary code exactly, identical edge_index with 0.000e+00 difference in
positions and velocities. That last check is the one case_05 had previously failed.

## 33. Case 05 shipping run with stationary ghosts and rebuilt data (17_ghost_zeros)

Every split rebuilt from the raw CSVs, including the 2,073-sphere cylinder, with stationary
ghosts, the one-sided cuboid reflection, and the corrected cylinder reflections. 250 epochs,
best validation 3.3324e-01 at the end of the schedule.

| rollout | metric | 17 rebuilt | 16_final stale data |
|---|---|---|---|
| case_06 cuboid | absolute pos | **4.20 mm** | 6.03 mm |
| | scaled pos | **0.6725** | 0.9642 |
| | scaled vel / angvel | **0.1028 / 0.1183** | 0.1157 / 0.1162 |
| rotating cylinder | absolute pos | **16.10 mm** | 151.3 mm |
| | scaled pos | **2.5755** | 24.2118 |
| | scaled vel | **0.0392** | 0.1633 |
| | scaled pos at final step | **2.6831** | 71.5930 |
| best validation | | **3.3324e-01** | 3.9410e-01 |

The cylinder is the striking one. Position error over the full 1998 steps drops by a factor of
9.4, and more importantly it stops diverging: 1.66 at step 600 and 2.58 averaged over the whole
rollout, ending at 2.68. The previous run grew from 7.57 at step 600 to 71.59 at the last step.
Velocity error falls by a factor of four. A zero-shot extrapolation to 2,073 spheres now holds
a bounded error for 20,000 forward passes.

Three things changed together here -- stationary ghosts, the |d| cylinder reflections, and the
dataset rebuild -- so the improvement cannot be attributed to any one of them from this run
alone. The training/validation mismatch documented in section 31 is the most likely dominant
factor for the cuboid numbers, since it made the earlier validation score meaningless; the
cylinder gain is plausibly the reflection fix, but that is not established.

### 33.1 Validation history

Best validation across the three case_05 attempts, all 250 epochs:

| run | dataset | best |
|---|---|---|
| 16_final | 2026-08-09, mirrored-ghost mismatch | 3.9410e-01 |
| 17 abandoned | 2026-08-09, same data | 3.9519e-01 |
| **17 rebuilt** | rebuilt 2026-08-13 | **3.3324e-01** |

The two runs on the stale dataset agree with each other to 0.3% and both sit 18% worse than
the rebuilt run, which is what a systematic data problem looks like rather than run-to-run
variance.

## 34. Final case_05 model: the external field

The external-force head was the dominant defect in case_05, and it took three corrections.

**What was wrong.** The head decoded a free 3-vector from the static node latent, added once
per sub-step. Every checkpoint learned a badly wrong field:

| checkpoint | learned acceleration (m/s^2) | error |
|---|---|---|
| legacy 6 Aug | [-3.544, -8.163, -1.574] | 4.221 |
| 20, legacy architecture | [-4.341, -5.963, -2.065] | 6.169 |
| 17_ghost_zeros | [-2.294, -11.445, +5.301] | 5.998 |
| measured from the data | [ 0.000, -9.829, 0.000] | -- |

The measurement is direct: spheres with no contacts in a frame accelerate under gravity alone,
giving (0, -9.829, 0) in training and (0, -9.796, 0) in validation across 1006 samples, with
the x and z components exactly zero and per-axis standard deviation 0.000.

An unconstrained 3-vector let the head absorb systematic error the contact model should have
explained. In the cuboid this is nearly invisible, since a spurious lateral push just presses
material into a nearby wall. In the drum, whose axis is 100 mm and unobstructed, the same
error integrated into a bed that marched 16 mm toward one end plate.

**The three corrections.**

1. Fix the direction. The axis is hard-coded to y in the model, so the x and z components are
   structurally zero rather than merely small.
2. Decode an acceleration, not a per-sub-step velocity change, and multiply by `sub_tstep`
   when applying it. The learned number is now a true m/s^2, independent of `num_msgs` and the
   timestep. With only the direction fixed and the old parameterisation, run 18 learned
   -23.47 m/s^2, worse in magnitude than the free-vector models.
3. Raise the learning rate. At 3e-5 training stalled by epoch 10 in two separate runs; at
   3e-4 it reached 2.7770e-01; at 1e-3 it reached 2.2577e-01 with 10 improvements.

**Result (24_extacc_200ep_lr1e3, 200 epochs, lr 1e-3).** The model recovers gravity on its own:

    learned  = [-0.0000, -9.6707, -0.0000] m/s^2
    measured = [ 0.0000, -9.8290,  0.0000] m/s^2
    error    = 0.158 m/s^2, 98.4% of gravity

Confirmed independently by integrating an isolated sphere: free fall is -9.6707 m/s^2. The
error falls from 4.2-13.6 m/s^2 to 0.158, a factor of 27 against the best previous checkpoint.

The value tracks whichever checkpoint validation selects, and is not monotone: intermediate
best checkpoints in this run read -10.78, -9.22 and -9.80 before the final -9.67. The figure
quoted here is the one in the shipped weights.

| metric | 24 final | 17_ghost_zeros |
|---|---|---|
| best validation | **2.2577e-01** | 3.3324e-01 |
| case_06 scaled pos MAE | **0.6298** (3.94 mm) | 0.9642 (6.03 mm) |
| drum scaled pos MAE | **0.9316** (5.82 mm) | 2.5755 (16.10 mm) |
| drum velocity / angular velocity | **0.0188 / 0.0390** | 0.0392 / 0.0902 |
| drum axial drift | within 2.5 mm | +16 mm, systematic |

### 34.1 What this run does not establish

The learning rate and the head parameterisation changed together between runs 18 and 24, so
their separate contributions are not isolated here. The physical argument for decoding an
acceleration stands on its own -- it removes a dependence on `num_msgs` and the timestep -- but
the validation gain cannot be attributed to it from this evidence.

`lambdaij_scaler` remains removed and softplus remains on the decoded inverse mass and inertia.
Earlier bisect arms suggested restoring the scaler helped, but run 24 reaches 2.2577e-01
against 2.2703e-01 for the legacy architecture at the same learning rate, so that gap closes
without it. The exact angular-momentum conservation is kept.

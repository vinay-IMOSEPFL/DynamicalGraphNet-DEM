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

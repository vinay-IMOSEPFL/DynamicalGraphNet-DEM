import os

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# Base directories
DATA_DIR = os.path.join(_HERE, "data")
DATASET_DIR = os.path.join(_HERE, "dataset")
SAVED_MODELS_DIR = os.path.join(_HERE, "saved_models")
RESULTS_DIR = os.path.join(_HERE, "results")

# Ensure required directories exist
for _d in (DATASET_DIR, SAVED_MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical Constants & Geometries
# ---------------------------------------------------------------------------
SPHERE_DIAMETER = 0.005
SPHERE_RADIUS = SPHERE_DIAMETER / 2.0
THRESHOLD = 1.25 * SPHERE_DIAMETER  # 1.25 * 0.005

# 1. Gravity Cuboid Setup (Training/Validation/Testing)
GEO_DATA_CUBOID = ((0.0, 0.0, 0.0), (0.03, 0.03, 0.03))
SAMPLE_TIME_STEP_CUBOID = 1e-4

# 2. Rotating Cylinder Setup (Extrapolation Rollout)
GEO_DATA_CYLINDER = ((-0.05, -0.048, 0.0), (0.05, 0.052, 0.1))
SAMPLE_TIME_STEP_CYLINDER = 1e-3  # Ground truth save interval for the cylinder case

# ---------------------------------------------------------------------------
# Model and training hyperparameters
# ---------------------------------------------------------------------------
MODEL_SETTINGS = {
    "batch_size": 64,
    "epochs": 200,
    "lr": 3e-4,
    "nf": 128,
    "n_layers": 2,
    "num_msgs": 5,
    "time_step": SAMPLE_TIME_STEP_CUBOID,
    "threshold": THRESHOLD,
    "node_in_f": 1,
    "use_ext_force": True,
}

# ---------------------------------------------------------------------------
# Data Splits (Heterogeneous Gravity ONLY)
# ---------------------------------------------------------------------------
# These paths map directly to the folder structure created by download_data.py
SPLIT_DIRS = {
    "train": [
        os.path.join("heterogeneous", "gravity", "training", "case_01"),
        os.path.join("heterogeneous", "gravity", "training", "case_02"),
        os.path.join("heterogeneous", "gravity", "training", "case_03"),
        os.path.join("heterogeneous", "gravity", "training", "case_04"),
        os.path.join("heterogeneous", "gravity", "training", "case_05"),
    ],
    "val": [
        os.path.join("heterogeneous", "gravity", "validation", "case_06")
    ],
    "test": [
        os.path.join("heterogeneous", "gravity", "extrapolation", "case_07")
    ],
    "cylinder": [
        os.path.join("heterogeneous", "gravity", "rotating_cylinder")
    ]
}

SEED = 100
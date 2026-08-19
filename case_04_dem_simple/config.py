# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

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
# Physical constants
# ---------------------------------------------------------------------------
GEO_DATA = ((0, 0, 0), (0.03, 0.03, 0.03))
SPHERE_DIAMETER = 0.005
SPHERE_RADIUS = SPHERE_DIAMETER / 2.0
THRESHOLD = 1.25 * SPHERE_DIAMETER
SAMPLE_TIME_STEP = 1e-4             # Ground-truth save interval, seconds

# ---------------------------------------------------------------------------
# Model and training hyperparameters
# ---------------------------------------------------------------------------
MODEL_SETTINGS = {
    "batch_size": 64,
    "epochs": 250,
    "lr": 3e-4,             # Paper: "a learning rate of 3 x 10^-4", Adam
    "nf": 128,              # Latent width
    "n_layers": 2,
    "num_msgs": 5,          # Integration sub-steps per sample step
    "time_step": SAMPLE_TIME_STEP,
    "threshold": THRESHOLD,
    "node_in_f": 1,
    # No external field in this case: the spheres are sealed in a cuboid with no
    # gravity, so the system is closed and its momentum is conserved.
    "use_ext_force": False,
}

# ---------------------------------------------------------------------------
# Data Splits
# ---------------------------------------------------------------------------
SPLIT_DIRS = {
    "train": ["CASE01", "CASE02", "CASE03", "CASE04", "CASE05"],
    "val": ["CASE06"],
    "test": ["CASE07"],
    "expt": ["1x"]
}
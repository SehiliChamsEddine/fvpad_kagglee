"""
Central configuration for the finger-vein PAD "mixed embeddings" project.

Every path, random seed and experiment-wide setting lives here, so that the
scripts and the notebook always use exactly the same values. If you need to
change something (e.g. where the data lives), change it here only.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Project root = the folder that contains the `fvpad` package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the three course archives were extracted. Override with the
# environment variable FVPAD_DATA_ROOT if your data lives somewhere else.
#   DATA_ROOT/real       <- Data_FV_Spoofing_WS2025_26.zip
#   DATA_ROOT/AE_Res     <- AE_Res-diff.zip
#   DATA_ROOT/VAE        <- VAE-diff.zip
DATA_ROOT = Path(os.environ.get("FVPAD_DATA_ROOT", PROJECT_ROOT / "data"))
REAL_ROOT = DATA_ROOT / "real"
SYNTH_ROOTS = {
    "AE_Res-diff": DATA_ROOT / "AE_Res",
    "VAE-diff": DATA_ROOT / "VAE",
}

WEIGHTS_DIR = PROJECT_ROOT / "weights"          # pretrained CNN weights
OUTPUT_DIR = PROJECT_ROOT / "outputs"           # everything we produce
MANIFEST_DIR = OUTPUT_DIR / "manifests"         # image lists + eligibility
FOLD_DIR = OUTPUT_DIR / "folds"                 # subject -> fold tables
FEATURE_DIR = OUTPUT_DIR / "features"           # cached CNN embeddings
RESULT_DIR = OUTPUT_DIR / "results"             # per-run experiment records
TABLE_DIR = OUTPUT_DIR / "tables"               # summary tables
FIGURE_DIR = OUTPUT_DIR / "figures"             # plots

# --------------------------------------------------------------------------
# Datasets and models
# --------------------------------------------------------------------------
DATASETS = ("IDIAP", "SCUT", "PLUS")

# The five feature extractors, named as in the teacher's earlier papers.
# The order here is also the order used when concatenating embeddings.
MODEL_NAMES = ("TransferNN", "VGGKNN", "TinyKNN", "MobileKNN", "SqueezeKNN")

SYNTH_FAMILIES = ("AE_Res-diff", "VAE-diff")

# --------------------------------------------------------------------------
# Protocol settings
# --------------------------------------------------------------------------
N_FOLDS = 5

# One master seed. Fold assignment and the random training-fold choices are
# derived from it deterministically, so every run reproduces the same splits.
SEED = 2026

# k-NN: candidate k values (odd, to avoid ties in a two-class vote). k is
# chosen per training set by an inner, subject-grouped cross-validation that
# only ever sees the training folds.
K_GRID = (1, 3, 5, 7, 9, 11, 15)
# Tie-break preference: k = 5 was used for TransferNN in the teacher's SAC
# paper. Among k values with the same (lowest) inner-validation ACER we take
# the one closest to DEFAULT_K, then the smaller one.
DEFAULT_K = 5
INNER_SPLITS = 5

# Feature extraction
BATCH_SIZE = 16
NUM_THREADS = max(1, (os.cpu_count() or 1))


def ensure_dirs() -> None:
    """Create all output folders (safe to call many times)."""
    for d in (WEIGHTS_DIR, OUTPUT_DIR, MANIFEST_DIR, FOLD_DIR, FEATURE_DIR,
              RESULT_DIR, TABLE_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

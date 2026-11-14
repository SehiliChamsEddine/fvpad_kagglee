"""
Normalising and combining ("mixing") embeddings of different CNNs.

Per-model L2 normalisation, then concatenation
----------------------------------------------
For an image, model m gives the raw embedding z_m. We use

    z_hat_m = z_m / (||z_m||_2 + eps)          (unit length)
    z_fused = [z_hat_A ; z_hat_B ; ...]         (concatenation)

Why L2 per model?
* The raw embeddings live on very different scales (mean norms on our data:
  VGG16 ~60, SqueezeNet ~70, EfficientNet/TinyNet/MobileNet ~13-15) and have
  different lengths (512 vs 1280). Concatenating raw vectors would let the
  large-norm networks dominate every Euclidean distance.
* After L2 normalisation every block has length 1, so each model contributes
  a squared distance in [0, 4] regardless of its dimension or scale - every
  CNN gets an equal vote in the fused distance.
* For a single model, Euclidean distance between unit vectors is a monotone
  function of the cosine similarity (||a-b||^2 = 2 - 2 cos(a,b)), the
  standard way of comparing CNN embeddings.
* It is computed per image, so it needs no statistics from other images:
  there is nothing to fit, and therefore no possibility of test-set leakage.

A useful consequence (used to make the experiments fast and exact):
for unit-length blocks the squared distance of the fused vectors is the SUM
of the per-model squared distances,

    ||z_fused(x) - z_fused(y)||^2 = sum_m (2 - 2 <z_hat_m(x), z_hat_m(y)>).

So we compute one similarity matrix per model and add them up for every
model combination, instead of re-building 31 concatenated feature matrices.
`check_block_decomposition` verifies this identity numerically.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import config

EPS = 1e-12


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + EPS)


def all_combinations(models=config.MODEL_NAMES) -> list[tuple[str, ...]]:
    """All 31 non-empty subsets of the five models (1, 2, 3, 4 and 5 models)."""
    return [c for r in range(1, len(models) + 1) for c in combinations(models, r)]


def combo_name(combo: tuple[str, ...]) -> str:
    return "+".join(combo)


def concat_normalized(blocks: dict[str, np.ndarray], combo: tuple[str, ...]) -> np.ndarray:
    """Explicit fused feature matrix (used for illustration and checks)."""
    return np.concatenate([l2_normalize(blocks[m]) for m in combo], axis=1)


def squared_distances(sims: dict[str, np.ndarray], combo: tuple[str, ...]) -> np.ndarray:
    """Squared Euclidean distances of the fused vectors from per-model
    cosine-similarity matrices (see the module docstring)."""
    total = sum(sims[m] for m in combo)
    return np.maximum(2.0 * len(combo) - 2.0 * total, 0.0)


def check_block_decomposition(blocks: dict[str, np.ndarray], combo: tuple[str, ...], n: int = 50) -> float:
    """Max abs difference between direct and decomposed squared distances."""
    Z = concat_normalized({m: b[:n] for m, b in blocks.items()}, combo)
    direct = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    sims = {m: l2_normalize(blocks[m][:n]) @ l2_normalize(blocks[m][:n]).T for m in combo}
    return float(np.abs(direct - squared_distances(sims, combo)).max())


def distance_share(blocks: dict[str, np.ndarray], normalize: bool, n_pairs: int = 20000,
                   seed: int = 0) -> dict[str, float]:
    """Average share of the total squared distance contributed by each model,
    for random image pairs - shows why normalisation is needed."""
    rng = np.random.default_rng(seed)
    n = next(iter(blocks.values())).shape[0]
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    parts = {}
    for m, X in blocks.items():
        Xm = l2_normalize(X) if normalize else X
        parts[m] = ((Xm[i] - Xm[j]) ** 2).sum(1)
    total = sum(parts.values()) + EPS
    return {m: float(np.mean(p / total)) for m, p in parts.items()}

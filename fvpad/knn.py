"""
k-nearest-neighbour classification with training-only selection of k.

The classifier works on a precomputed distance matrix (rows = query images,
columns = training images). A query image is labelled spoof if the majority
of its k nearest training images are spoofs. k is odd, so a two-class vote
can never tie; equal distances are resolved by the (fixed) training order.
This is exactly what scikit-learn's KNeighborsClassifier(metric="precomputed",
weights="uniform") does - `tests` in the notebook confirm identical output.

Choosing k without touching the test fold
-----------------------------------------
For a given training set we run an *inner* cross-validation inside it:
the training subjects are split into INNER_SPLITS groups (GroupKFold, so all
images of a subject - including synthetic images linked to that subject -
stay in the same inner fold). For every candidate k we predict each inner
validation fold from the remaining inner folds and accumulate one confusion
matrix. The k with the lowest inner ACER is chosen; ties are broken towards
k = DEFAULT_K (the value used in the teacher's SAC paper), then towards the
smaller k. Only then is the held-out test fold classified - once.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from . import config
from .metrics import rates


def neighbour_votes(D: np.ndarray, y_train: np.ndarray, k_max: int) -> np.ndarray:
    """Cumulative spoof votes: votes[i, k-1] = number of spoofs among the k
    nearest training images of query i (k = 1 .. k_max)."""
    k_max = min(k_max, D.shape[1])
    # indices of the k_max smallest distances, then sorted (stable) by distance
    part = np.argpartition(D, k_max - 1, axis=1)[:, :k_max]
    order = np.take_along_axis(D, part, axis=1).argsort(axis=1, kind="stable")
    nn = np.take_along_axis(part, order, axis=1)
    return np.cumsum(y_train[nn], axis=1)


def predict(D: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    votes = neighbour_votes(D, y_train, k)[:, min(k, D.shape[1]) - 1]
    return (votes > min(k, D.shape[1]) / 2).astype(int)


def select_k(D_train: np.ndarray, y: np.ndarray, groups: np.ndarray,
             k_grid=config.K_GRID, n_splits: int = config.INNER_SPLITS) -> tuple[int, dict[int, float]]:
    """Inner subject-grouped CV on the training set only. Returns (best_k, {k: inner ACER})."""
    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    cm = {k: np.zeros(4, dtype=int) for k in k_grid}           # TP, FN, FP, TN
    for tr, va in GroupKFold(n_splits=n_splits).split(D_train, y, groups):
        votes = neighbour_votes(D_train[np.ix_(va, tr)], y[tr], max(k_grid))
        yv = y[va]
        for k in k_grid:
            kk = min(k, len(tr))
            pred = (votes[:, kk - 1] > kk / 2).astype(int)
            cm[k] += [((yv == 1) & (pred == 1)).sum(), ((yv == 1) & (pred == 0)).sum(),
                      ((yv == 0) & (pred == 1)).sum(), ((yv == 0) & (pred == 0)).sum()]
    inner = {k: rates(*map(int, cm[k]))["ACER"] for k in k_grid}
    score = {k: round(v, 9) if np.isfinite(v) else np.inf for k, v in inner.items()}
    best = min(k_grid, key=lambda k: (score[k], abs(k - config.DEFAULT_K), k))
    return best, inner

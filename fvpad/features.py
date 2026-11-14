"""
Feature (embedding) extraction with an on-disk cache.

For every model and dataset the embeddings are stored in
    outputs/features/<dataset>/<model>.npz   (arrays: image_id, X)
Each row of X is the *raw* GAP embedding of one image, before any
normalisation. Extraction is incremental: only images that are not yet in
the cache are processed, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .models import MODEL_SPECS, load_gray, load_model, preprocess


def cache_path(dataset: str, model: str) -> Path:
    return config.FEATURE_DIR / dataset / f"{model}.npz"


def load_cache(dataset: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    p = cache_path(dataset, model)
    if not p.exists():
        return np.array([], dtype=object), np.zeros((0, MODEL_SPECS[model].embed_dim), np.float32)
    z = np.load(p, allow_pickle=True)
    return z["image_id"], z["X"]


def _save_cache(dataset: str, model: str, ids: np.ndarray, X: np.ndarray) -> None:
    p = cache_path(dataset, model)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, image_id=ids.astype(object), X=X.astype(np.float32))
    tmp.replace(p)


def _device() -> torch.device:
    """GPU if available (e.g. on Kaggle/Colab), otherwise CPU. Results are the
    same up to floating-point rounding (~1e-6)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def extract(manifest: pd.DataFrame, models: tuple[str, ...] = config.MODEL_NAMES,
            batch_size: int = config.BATCH_SIZE, save_every: int = 1000, verbose: bool = True) -> None:
    """Compute and cache embeddings for all images in `manifest` that are not cached yet."""
    torch.set_num_threads(config.NUM_THREADS)
    device = _device()
    pool = ThreadPoolExecutor(max_workers=config.NUM_THREADS)     # parallel image decoding
    for model_name in models:
        spec = MODEL_SPECS[model_name]
        net = None
        for ds, g in manifest.groupby("dataset", sort=False):
            ids, X = load_cache(ds, model_name)
            have = set(ids.tolist())
            todo = g[~g.image_id.isin(have)]
            if todo.empty:
                if verbose:
                    print(f"[{model_name}/{ds}] all {len(g)} embeddings cached")
                continue
            if net is None:
                net = load_model(model_name).to(device)
            t0 = time.time()
            new_ids, new_X = [], []
            paths, todo_ids = todo.path.tolist(), todo.image_id.tolist()
            for i in range(0, len(todo), batch_size):
                batch = np.stack(list(pool.map(lambda p: preprocess(load_gray(config.DATA_ROOT / p), spec),
                                               paths[i:i + batch_size])))
                new_X.append(net(torch.from_numpy(batch).to(device)).float().cpu().numpy())
                new_ids.extend(todo_ids[i:i + batch_size])
                done = i + batch_size
                if done % save_every < batch_size or done >= len(todo):
                    ids = np.concatenate([ids, np.array(new_ids, dtype=object)])
                    X = np.concatenate([X, *new_X]) if new_X else X
                    _save_cache(ds, model_name, ids, X)
                    new_ids, new_X = [], []
                    if verbose:
                        rate = min(done, len(todo)) / (time.time() - t0)
                        print(f"[{model_name}/{ds}] {min(done, len(todo))}/{len(todo)} new "
                              f"({rate:.1f} img/s, {device})", flush=True)


def get_matrix(dataset: str, model: str, image_ids) -> np.ndarray:
    """Raw embeddings for the given image ids (in that order)."""
    ids, X = load_cache(dataset, model)
    pos = {k: i for i, k in enumerate(ids.tolist())}
    missing = [k for k in image_ids if k not in pos]
    if missing:
        raise KeyError(f"{len(missing)} images of {dataset} have no {model} embedding, e.g. {missing[:3]}")
    return X[[pos[k] for k in image_ids]]

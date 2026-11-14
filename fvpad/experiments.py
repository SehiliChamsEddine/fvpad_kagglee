"""
Running Task 1 (BA, S1, S2) and Task 2 (S3, S4, S5) for every dataset and
every one of the 31 CNN combinations.

One *run* = (dataset, step, synthetic family, CNN combination, test fold).

    training set : real images of the step's real training folds
                   + selected synthetic images of its synthetic training folds
    test set     : real genuine + real spoof images of the held-out fold
                   (never synthetic, never seen during training or k-selection)

Each run is written as one row of outputs/results/runs.csv with everything
needed to trace it (folds, data composition, k, confusion counts, APCER,
BPCER, ACER). The per-image test predictions are stored as well
(outputs/results/predictions_task<t>_<dataset>.npz, key = run_id).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, features, knn
from .folds import STEPS_TASK1, STEPS_TASK2, parse_folds
from .fusion import all_combinations, combo_name, l2_normalize, squared_distances
from .metrics import pad_metrics

RUNS_FILE = config.RESULT_DIR / "runs.csv"            # Task 1 + Task 2 combined


def runs_file(task: int):
    return config.RESULT_DIR / f"runs_task{task}.csv"
KNN_DESCRIPTION = "uniform-vote kNN, Euclidean distance on concatenated per-model L2-normalised embeddings"


def dataset_table(tables: dict[str, pd.DataFrame], ds: str) -> pd.DataFrame:
    """All images of one dataset that take part in experiments, with their fold."""
    fmap = {(r.dataset, r.subject): r.fold for r in tables["folds"].itertuples()}
    real = tables["real"]
    real = real[(real.dataset == ds) & real.eligible].copy()
    real["fold"] = [fmap[(d, s)] for d, s in zip(real.dataset, real.subject)]
    synth = tables["synth"]
    synth = synth[(synth.dataset == ds) & synth.selected].copy()
    cols = ["image_id", "source", "family", "label", "y", "subject", "fold"]
    return pd.concat([real[cols], synth[cols]], ignore_index=True)


def load_normalized_blocks(df: pd.DataFrame, ds: str) -> dict[str, np.ndarray]:
    """L2-normalised embeddings of every image in df, one matrix per model."""
    return {m: l2_normalize(features.get_matrix(ds, m, df.image_id.tolist())) for m in config.MODEL_NAMES}


def training_indices(df: pd.DataFrame, real_folds: list[int], synth_folds: list[int], family: str) -> np.ndarray:
    is_real = (df.source == "real") & df.fold.isin(real_folds)
    is_syn = (df.source == "synthetic") & (df.family == family) & df.fold.isin(synth_folds)
    return np.flatnonzero(is_real | is_syn)


def task_jobs(task: int) -> list[tuple[str, str]]:
    """(step, synthetic family) pairs of a task; "-" means no synthetic data."""
    if task == 1:
        return [(s, "-") for s in STEPS_TASK1]
    return [(s, fam) for fam in config.SYNTH_FAMILIES for s in STEPS_TASK2]


def run_dataset(tables: dict[str, pd.DataFrame], ds: str, task: int, combos=None, verbose: bool = True):
    """All runs of one dataset and task. Returns (list of result rows, {run_id: predictions})."""
    combos = combos or all_combinations()
    df = dataset_table(tables, ds)
    blocks = load_normalized_blocks(df, ds)
    y = df.y.to_numpy()
    groups = df.subject.to_numpy()
    sel = tables["selections"]
    rows, preds = [], {}
    t0 = time.time()
    for test_fold in range(1, config.N_FOLDS + 1):
        te = np.flatnonzero((df.source == "real") & (df.fold == test_fold))
        for step, family in task_jobs(task):
            choice = sel[(sel.dataset == ds) & (sel.test_fold == test_fold) & (sel.step == step)].iloc[0]
            real_f, syn_f = parse_folds(choice.real_train_folds), parse_folds(choice.synth_train_folds)
            tr = training_indices(df, real_f, syn_f, family)
            assert test_fold not in real_f + syn_f
            assert not set(groups[tr]) & set(groups[te]), "subject overlap between training and test"
            # per-model cosine similarities, computed once, summed for each combination
            sim_tr = {m: blocks[m][tr] @ blocks[m][tr].T for m in config.MODEL_NAMES}
            sim_te = {m: blocks[m][te] @ blocks[m][tr].T for m in config.MODEL_NAMES}
            src, lab = df.source.to_numpy()[tr], y[tr]
            comp = dict(n_train_real_genuine=int(((src == "real") & (lab == 0)).sum()),
                        n_train_real_spoof=int(((src == "real") & (lab == 1)).sum()),
                        n_train_synth_genuine=int(((src == "synthetic") & (lab == 0)).sum()),
                        n_train_synth_spoof=int(((src == "synthetic") & (lab == 1)).sum()),
                        n_test_genuine=int((y[te] == 0).sum()), n_test_spoof=int((y[te] == 1).sum()))
            for combo in combos:
                D_tr = squared_distances(sim_tr, combo)
                D_te = squared_distances(sim_te, combo)
                k, inner = knn.select_k(D_tr, y[tr], groups[tr])
                pred = knn.predict(D_te, y[tr], k)
                run_id = f"{ds}|{step}|{family}|{combo_name(combo)}|{test_fold}"
                preds[run_id] = pred.astype(np.uint8)
                rows.append(dict(
                    run_id=run_id, dataset=ds, task=1 if step in STEPS_TASK1 else 2, step=step,
                    synth_family=family, combo=combo_name(combo), n_models=len(combo),
                    test_fold=test_fold, real_train_folds=choice.real_train_folds,
                    synth_train_folds=choice.synth_train_folds, **comp,
                    k=k, inner_acer=inner[k], knn=KNN_DESCRIPTION,
                    **pad_metrics(y[te], pred)))
        if verbose:
            print(f"[{ds}, Task {task}] test fold {test_fold} done ({time.time() - t0:.0f}s)", flush=True)
    return rows, preds


def run_task(tables: dict[str, pd.DataFrame], task: int, datasets=config.DATASETS,
             verbose: bool = True) -> pd.DataFrame:
    """Run one task for all datasets; saves outputs/results/runs_task<task>.csv
    and the per-image predictions."""
    config.ensure_dirs()
    all_rows = []
    for ds in datasets:
        rows, preds = run_dataset(tables, ds, task, verbose=verbose)
        all_rows.extend(rows)
        np.savez_compressed(config.RESULT_DIR / f"predictions_task{task}_{ds}.npz", **preds)
    runs = pd.DataFrame(all_rows)
    runs.to_csv(runs_file(task), index=False)
    return runs


def combine_tasks() -> pd.DataFrame:
    """Concatenate the Task 1 and Task 2 records into runs.csv."""
    runs = pd.concat([pd.read_csv(runs_file(t)) for t in (1, 2)], ignore_index=True)
    runs.to_csv(RUNS_FILE, index=False)
    return runs


def run_all(tables: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    run_task(tables, 1, verbose=verbose)
    run_task(tables, 2, verbose=verbose)
    return combine_tasks()


def test_image_ids(tables: dict[str, pd.DataFrame], ds: str, test_fold: int) -> pd.DataFrame:
    """The test images of a run, in the order of the stored predictions."""
    df = dataset_table(tables, ds)
    return df[(df.source == "real") & (df.fold == test_fold)].reset_index(drop=True)

"""
The pipeline stages. Both `scripts/run_pipeline.py` and the notebook call
exactly these functions, so they run the same procedure.

    prepare()      manifests, eligibility, synthetic attribution, folds, random choices
    extract()      CNN embeddings for every image used (cached)
    run()          all Task 1 and Task 2 experiments (see experiments.py)
    tables()       summary tables and figures (see results.py)
"""
from __future__ import annotations

import pandas as pd

from . import config, datasets, folds as folds_mod

FILES = {
    "real": config.MANIFEST_DIR / "real_manifest.csv",
    "synth": config.MANIFEST_DIR / "synthetic_manifest.csv",
    "thresholds": config.MANIFEST_DIR / "synthetic_attribution_thresholds.csv",
    "folds": config.FOLD_DIR / "subject_folds.csv",
    "selections": config.FOLD_DIR / "fold_selections.csv",
    "fold_sizes": config.FOLD_DIR / "fold_sizes.csv",
}

_DTYPES = {"subject": str, "finger": str, "orig_split": str, "pair_key": str, "nn_real_id": str}


def prepare(force: bool = False, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Build (or load) every table that defines the data and the protocol."""
    config.ensure_dirs()
    if not force and all(p.exists() for p in FILES.values()):
        out = {k: pd.read_csv(p, dtype=_DTYPES, keep_default_na=False) for k, p in FILES.items()}
        out["folds"]["subject"] = out["folds"]["subject"].astype(str)
        if verbose:
            print("Loaded existing manifests and folds from", config.OUTPUT_DIR)
        return out

    if verbose:
        print("Scanning real data ...")
    real = datasets.build_real_manifest()
    if verbose:
        print("Scanning synthetic data and attributing subjects (takes a few minutes) ...")
    synth = pd.concat([datasets.scan_synthetic(fam, root) for fam, root in config.SYNTH_ROOTS.items()],
                      ignore_index=True)
    synth, thresholds = datasets.attribute_synthetic(real, synth)

    subject_folds = folds_mod.assign_folds(real)
    selections = folds_mod.draw_fold_selections()
    synth = folds_mod.select_synthetic_folds(synth, real, subject_folds)
    sizes = folds_mod.fold_size_table(real, synth, subject_folds)

    out = dict(real=real, synth=synth, thresholds=thresholds, folds=subject_folds,
               selections=selections, fold_sizes=sizes)
    for k, df in out.items():
        df.to_csv(FILES[k], index=False)
    if verbose:
        print("Saved manifests and folds to", config.OUTPUT_DIR)
    return out


def images_to_embed(tables: dict[str, pd.DataFrame], all_synthetic: bool = False) -> pd.DataFrame:
    """Every image that takes part in an experiment: eligible real images and
    the selected synthetic images (or all synthetic images if all_synthetic)."""
    real = tables["real"]
    synth = tables["synth"]
    cols = ["dataset", "image_id", "path"]
    syn = synth if all_synthetic else synth[synth.selected]
    return pd.concat([real[real.eligible][cols], syn[cols]], ignore_index=True)


def extract(tables: dict[str, pd.DataFrame], models=config.MODEL_NAMES, all_synthetic: bool = False,
            verbose: bool = True) -> None:
    from . import features
    features.extract(images_to_embed(tables, all_synthetic), models=models, verbose=verbose)


def run(tables: dict[str, pd.DataFrame], verbose: bool = True) -> pd.DataFrame:
    from . import experiments
    return experiments.run_all(tables, verbose=verbose)


def tables(tables: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    from . import experiments, results
    runs = pd.read_csv(experiments.RUNS_FILE)
    return results.write_all(runs)

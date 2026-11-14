"""
Subject-disjoint five-fold partition, the random training-fold choices for
every experimental step, and the "synthetic version" of each fold.

Everything here is driven by `config.SEED`, so re-running the code always
produces the same folds and the same random choices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

STEPS_TASK1 = ("BA", "S1", "S2")
STEPS_TASK2 = ("S3", "S4", "S5")


def _rng(*keys: int) -> np.random.Generator:
    """Independent, reproducible random stream for a given purpose."""
    return np.random.default_rng([config.SEED, *keys])


# --------------------------------------------------------------------------
# 1. Subject -> fold
# --------------------------------------------------------------------------
def assign_folds(real: pd.DataFrame) -> pd.DataFrame:
    """Split the eligible subjects of each dataset into N_FOLDS folds.

    * A subject (all its fingers, sessions, genuine and spoof images) goes to
      exactly one fold.
    * Subjects are shuffled with a fixed seed, then placed one by one (largest
      first) into the fold that currently holds the fewest images, so folds are
      as equal as possible in images and in subjects.

    Returns a table: dataset, subject, fold (1..5), n_images.
    """
    rows = []
    for d_idx, ds in enumerate(config.DATASETS):
        e = real[(real.dataset == ds) & real.eligible]
        counts = e.groupby("subject").size()
        subjects = counts.index.to_numpy()
        order = _rng(1, d_idx).permutation(len(subjects))
        subjects = subjects[order]
        # stable sort by image count (desc) keeps the random order among equals
        subjects = sorted(subjects, key=lambda s: -counts[s])
        load = np.zeros(config.N_FOLDS, dtype=int)
        nsub = np.zeros(config.N_FOLDS, dtype=int)
        for s in subjects:
            f = int(np.lexsort((nsub, load))[0])        # fewest images, then fewest subjects
            load[f] += counts[s]
            nsub[f] += 1
            rows.append(dict(dataset=ds, subject=s, fold=f + 1, n_images=int(counts[s])))
    return pd.DataFrame(rows).sort_values(["dataset", "fold", "subject"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. Random training-fold choices for every step
# --------------------------------------------------------------------------
def draw_fold_selections() -> pd.DataFrame:
    """For every dataset and test fold draw ONE random order (a, b, c, d) of
    the four remaining folds and derive all steps from it:

        BA : real {a, b, c, d}
        S1 : real {a, b}                    (2/5 of the data)
        S2 : real {a}                       (1/5)
        S3 : real {a}       + synthetic {b}           (same size as S1)
        S4 : real {a}       + synthetic {b, c, d}     (same size as BA)
        S5 :                  synthetic {a, b, c, d}  (same size as BA)

    The choices are random (as required) and nested: S2's fold is one of
    S1's folds, and S3 replaces S1's second fold by its synthetic version -
    exactly the "add the synthetic samples of the removed fold" wording of the
    official instructions. Using one draw for all CNN combinations and both
    synthetic families makes every comparison paired (same training folds).
    """
    rows = []
    for d_idx, ds in enumerate(config.DATASETS):
        for test in range(1, config.N_FOLDS + 1):
            rest = [f for f in range(1, config.N_FOLDS + 1) if f != test]
            a, b, c, d = (int(x) for x in _rng(2, d_idx, test).permutation(rest))
            plan = {
                "BA": ([a, b, c, d], []),
                "S1": ([a, b], []),
                "S2": ([a], []),
                "S3": ([a], [b]),
                "S4": ([a], [b, c, d]),
                "S5": ([], [a, b, c, d]),
            }
            for step, (real_f, syn_f) in plan.items():
                rows.append(dict(dataset=ds, test_fold=test, step=step,
                                 real_train_folds=" ".join(map(str, sorted(real_f))),
                                 synth_train_folds=" ".join(map(str, sorted(syn_f))),
                                 random_order=f"{a} {b} {c} {d}"))
    return pd.DataFrame(rows)


def parse_folds(s: str) -> list[int]:
    return [int(x) for x in str(s).split()] if isinstance(s, str) and s.strip() else []


# --------------------------------------------------------------------------
# 3. Synthetic version of each fold
# --------------------------------------------------------------------------
def select_synthetic_folds(synth: pd.DataFrame, real: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Give every synthetic image the fold of its attributed subject, and pick
    the images that form the "synthetic version" of each fold.

    Per (dataset, family, fold, class) we draw - at random, with a fixed seed -
    as many synthetic images as the real fold has images of that class, so a
    synthetic fold can stand in for a real fold of the same size.

    Images whose subject link is confident (`attributed`, i.e. near-copies of
    a real image of that subject) are drawn first; only if there are not
    enough of them is the fold topped up with the remaining images assigned to
    it by nearest-subject similarity. For IDIAP and PLUS this means every
    selected synthetic image is a confident copy of a subject of that fold.
    Genuine and spoof counts are kept equal. If fewer images than needed exist
    (SCUT) we use as many as the smaller class allows; the true numbers are
    recorded with every experiment.

    Returns synth with the extra columns `fold` and `selected`.
    """
    synth = synth.copy()
    fmap = {(r.dataset, r.subject): r.fold for r in folds.itertuples()}
    synth["fold"] = [fmap[(d, s)] for d, s in zip(synth.dataset, synth.subject)]
    synth["selected"] = False
    real_e = real[real.eligible].copy()
    real_e["fold"] = [fmap[(d, s)] for d, s in zip(real_e.dataset, real_e.subject)]
    need = real_e.groupby(["dataset", "fold", "label"]).size()
    for d_idx, ds in enumerate(config.DATASETS):
        for f_idx, fam in enumerate(config.SYNTH_FAMILIES):
            for fold in range(1, config.N_FOLDS + 1):
                base = (synth.dataset == ds) & (synth.family == fam) & (synth.fold == fold)
                # same number of genuine and spoof images (balanced classes)
                n = min(int(need[(ds, fold, "genuine")]), int(need[(ds, fold, "spoof")]),
                        int((base & (synth.label == "genuine")).sum()),
                        int((base & (synth.label == "spoof")).sum()))
                for l_idx, label in enumerate(("genuine", "spoof")):
                    in_fold = base & (synth.label == label)
                    rng = _rng(3, d_idx, f_idx, fold, l_idx)
                    pick = []
                    for confident in (True, False):          # confident links first
                        pool = np.asarray(synth.index[in_fold & (synth.attributed == confident)])
                        take = min(n - len(pick), len(pool))
                        pick.extend(rng.choice(pool, size=take, replace=False).tolist())
                    synth.loc[pick, "selected"] = True
    return synth


def fold_size_table(real: pd.DataFrame, synth: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Images per fold: real genuine/spoof and selected synthetic genuine/spoof."""
    fmap = {(r.dataset, r.subject): r.fold for r in folds.itertuples()}
    r = real[real.eligible].assign(fold=lambda d: [fmap[k] for k in zip(d.dataset, d.subject)])
    t = r.pivot_table(index=["dataset", "fold"], columns="label", values="image_id", aggfunc="count")
    t.columns = [f"real_{c}" for c in t.columns]
    t["subjects"] = folds.groupby(["dataset", "fold"]).size()
    sel = synth[synth.selected]
    s = sel.pivot_table(index=["dataset", "fold"], columns=["family", "label"],
                        values="image_id", aggfunc="count")
    s.columns = [f"{fam}_{lab}" for fam, lab in s.columns]
    a = sel.pivot_table(index=["dataset", "fold"], columns="family", values="attributed", aggfunc="mean")
    a.columns = [f"{fam}_share_confident" for fam in a.columns]
    return t.join(s).join(a.round(3)).reset_index()

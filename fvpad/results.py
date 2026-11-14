"""
Turning the per-run records into summary tables.

Aggregation over the five test folds
------------------------------------
Every eligible real image is tested exactly once (in the run where its fold
is the test fold). We therefore add up the confusion counts of the five runs
and compute APCER / BPCER / ACER from the pooled counts ("pooled"). The mean
and standard deviation of the five per-fold ACER values are reported as well,
to show how much the result varies between folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .fusion import all_combinations, combo_name
from .metrics import rates

SETTINGS = [("BA", "-"), ("S1", "-"), ("S2", "-"),
            ("S3", "AE_Res-diff"), ("S4", "AE_Res-diff"), ("S5", "AE_Res-diff"),
            ("S3", "VAE-diff"), ("S4", "VAE-diff"), ("S5", "VAE-diff")]
SHORT = {"-": "", "AE_Res-diff": "AE", "VAE-diff": "VAE"}
COMBO_ORDER = [combo_name(c) for c in all_combinations()]
ABBR = {"TransferNN": "Eff", "VGGKNN": "VGG", "TinyKNN": "Tiny", "MobileKNN": "Mob", "SqueezeKNN": "Sq"}


def setting_label(step: str, family: str) -> str:
    return step if family == "-" else f"{step}-{SHORT[family]}"


def short_combo(name: str) -> str:
    return "+".join(ABBR[m] for m in name.split("+"))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "task", "step", "synth_family", "combo", "n_models"]
    g = runs.groupby(keys, sort=False)
    s = g[["TP", "FN", "FP", "TN"]].sum()
    pooled = pd.DataFrame([rates(*r) for r in s.itertuples(index=False)], index=s.index)
    out = s.join(pooled)
    out["ACER_fold_mean"] = g["ACER"].mean()
    out["ACER_fold_std"] = g["ACER"].std(ddof=0)
    out["k_per_fold"] = g["k"].apply(lambda v: " ".join(map(str, v)))
    out = out.reset_index()
    out["setting"] = [setting_label(a, b) for a, b in zip(out.step, out.synth_family)]
    return out


def acer_matrix(summary: pd.DataFrame, dataset: str, metric: str = "ACER") -> pd.DataFrame:
    """Rows = 31 CNN combinations, columns = the 9 settings."""
    d = summary[summary.dataset == dataset]
    m = d.pivot(index="combo", columns="setting", values=metric)
    cols = [setting_label(s, f) for s, f in SETTINGS]
    return m.loc[COMBO_ORDER, cols]


def step_table(summary: pd.DataFrame, dataset: str, step: str, family: str = "-") -> pd.DataFrame:
    """APCER / BPCER / ACER (pooled) and fold mean +- std of ACER for every combination."""
    d = summary[(summary.dataset == dataset) & (summary.step == step) & (summary.synth_family == family)]
    d = d.set_index("combo").loc[COMBO_ORDER]
    return d[["n_models", "APCER", "BPCER", "ACER", "ACER_fold_mean", "ACER_fold_std", "k_per_fold"]]


def step_table_wide(summary: pd.DataFrame, step: str, family: str = "-") -> pd.DataFrame:
    """One setting, all three datasets side by side: rows = 31 combinations,
    columns = (dataset, APCER/BPCER/ACER), pooled over the five test folds."""
    parts = {}
    for ds in config.DATASETS:
        t = step_table(summary, ds, step, family)[["APCER", "BPCER", "ACER"]]
        parts[ds] = t
    wide = pd.concat(parts, axis=1)
    wide.index = [short_combo(c) for c in wide.index]
    wide.index.name = f"{setting_label(step, family)}: CNN combination"
    return wide


def single_vs_fusion(summary: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and setting: best/mean single model vs. best/mean fusion and the 5-model fusion."""
    rows = []
    for ds in config.DATASETS:
        for step, fam in SETTINGS:
            d = summary[(summary.dataset == ds) & (summary.step == step) & (summary.synth_family == fam)]
            single, fused = d[d.n_models == 1], d[d.n_models > 1]
            bs, bf = single.loc[single.ACER.idxmin()], fused.loc[fused.ACER.idxmin()]
            rows.append(dict(
                dataset=ds, setting=setting_label(step, fam),
                best_single=short_combo(bs.combo), best_single_ACER=bs.ACER,
                mean_single_ACER=single.ACER.mean(),
                best_fusion=short_combo(bf.combo), best_fusion_ACER=bf.ACER,
                mean_fusion_ACER=fused.ACER.mean(),
                all5_ACER=float(d[d.n_models == 5].ACER.iloc[0]),
                fusions_beating_best_single=int((fused.ACER < bs.ACER).sum()),
                n_fusions=len(fused)))
    return pd.DataFrame(rows)


def by_n_models(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean pooled ACER over all combinations with the same number of models."""
    t = summary.groupby(["dataset", "setting", "n_models"]).ACER.mean().unstack("n_models")
    order = [setting_label(s, f) for s, f in SETTINGS]
    return t.reindex([(ds, s) for ds in config.DATASETS for s in order])


def model_contribution(summary: pd.DataFrame) -> pd.DataFrame:
    """For every model: mean ACER of the combinations that contain it minus the
    mean ACER of those that do not (negative = adding the model helps)."""
    rows = []
    for (ds, setting), d in summary.groupby(["dataset", "setting"], sort=False):
        for m in config.MODEL_NAMES:
            has = d.combo.str.split("+").apply(lambda c: m in c)
            rows.append(dict(dataset=ds, setting=setting, model=m,
                             delta_ACER=d[has].ACER.mean() - d[~has].ACER.mean()))
    t = pd.DataFrame(rows).pivot_table(index=["dataset", "setting"], columns="model", values="delta_ACER")
    order = [setting_label(s, f) for s, f in SETTINGS]
    return t.reindex([(ds, s) for ds in config.DATASETS for s in order])[list(config.MODEL_NAMES)]


def k_usage(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.groupby(["dataset", "step"]).k.value_counts().unstack(fill_value=0)


def write_all(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Write every summary table as CSV (+ one Excel workbook) to outputs/tables."""
    config.ensure_dirs()
    summary = summarize(runs)
    out = {"summary_all": summary, "single_vs_fusion": single_vs_fusion(summary),
           "acer_by_n_models": by_n_models(summary).reset_index(),
           "model_contribution": model_contribution(summary).reset_index(),
           "k_usage": k_usage(runs).reset_index()}
    for ds in config.DATASETS:
        for metric in ("ACER", "APCER", "BPCER"):
            out[f"{ds}_{metric}_matrix"] = acer_matrix(summary, ds, metric).reset_index()
    for name, df in out.items():
        df.to_csv(config.TABLE_DIR / f"{name}.csv", index=False, float_format="%.2f")
    try:
        with pd.ExcelWriter(config.TABLE_DIR / "all_results.xlsx") as xw:
            for name, df in out.items():
                df.to_excel(xw, sheet_name=name[:31], index=False)
            runs.to_excel(xw, sheet_name="runs", index=False)
    except ImportError:
        pass
    return out

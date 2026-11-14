"""
Figures for the notebook and the reports (static PNGs, saved to outputs/figures).

Colours follow one fixed categorical order (blue, orange, aqua, yellow,
magenta, green, ...) so a given series always has the same colour, and a
single-hue blue ramp for magnitudes (heatmaps).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from . import config  # noqa: E402
from .results import SETTINGS, acer_matrix, setting_label, short_combo  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, name: str):
    config.ensure_dirs()
    path = config.FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _img(rel_path: str) -> np.ndarray:
    with Image.open(config.DATA_ROOT / rel_path) as im:
        return np.asarray(im.convert("L"))


# --------------------------------------------------------------------------
# Data figures
# --------------------------------------------------------------------------
def example_images(tables: dict[str, pd.DataFrame], seed: int = 0):
    """One example per dataset and type: real genuine/spoof, AE_Res-diff and VAE-diff genuine/spoof."""
    real, synth = tables["real"], tables["synth"]
    cols = [("real", "real", "genuine"), ("real", "real", "spoof"),
            ("synthetic", "AE_Res-diff", "genuine"), ("synthetic", "AE_Res-diff", "spoof"),
            ("synthetic", "VAE-diff", "genuine"), ("synthetic", "VAE-diff", "spoof")]
    titles = ["real genuine", "real spoof", "AE_Res genuine", "AE_Res spoof", "VAE genuine", "VAE spoof"]
    fig, axes = plt.subplots(3, 6, figsize=(13, 7.2), gridspec_kw=dict(height_ratios=[1, 1.6, 1]))
    rng = np.random.default_rng(seed)
    for r, ds in enumerate(config.DATASETS):
        for c, (src, fam, lab) in enumerate(cols):
            pool = real[(real.dataset == ds) & real.eligible & (real.label == lab)] if src == "real" else \
                synth[(synth.dataset == ds) & (synth.family == fam) & (synth.label == lab) & synth.selected]
            row = pool.iloc[int(rng.integers(len(pool)))]
            ax = axes[r, c]
            ax.imshow(_img(row.path), cmap="gray", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c])
            if c == 0:
                ax.set_ylabel(ds, fontsize=11, fontweight="bold", color=INK)
    fig.suptitle("Example full images (all shown at the same panel size; native sizes differ)", color=INK2)
    return _save(fig, "example_images.png")


def attribution_histograms(tables: dict[str, pd.DataFrame]):
    """Similarity of every synthetic image to its most similar real image of
    the same class, against the highest similarity between two *different*
    real subjects (dashed line, tau)."""
    synth, thr = tables["synth"], tables["thresholds"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharey="row")
    bins = np.linspace(0.6, 1.0, 81)
    for c, ds in enumerate(config.DATASETS):
        for r, lab in enumerate(("genuine", "spoof")):
            ax = axes[r, c]
            for i, fam in enumerate(config.SYNTH_FAMILIES):
                v = synth[(synth.dataset == ds) & (synth.family == fam) & (synth.label == lab)].nn_corr
                ax.hist(np.clip(v, 0.6, 1.0), bins=bins, histtype="step", linewidth=2, color=SERIES[i], label=fam)
            tau = float(thr[(thr.dataset == ds) & (thr.label == lab)].tau.iloc[0])
            ax.axvline(tau, color=INK2, linestyle="--", linewidth=1.2)
            ax.text(tau - 0.004, ax.get_ylim()[1] * 0.92, f"tau={tau:.3f}", color=INK2, fontsize=8, ha="right")
            ax.set_title(f"{ds} - synthetic {lab}")
            if r == 1:
                ax.set_xlabel("correlation with most similar real image")
            if c == 0:
                ax.set_ylabel("synthetic images")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Right of the dashed line: closer to one real image than any two different subjects ever are "
                 "(near-copy, subject link confident)", color=INK2)
    fig.tight_layout()
    return _save(fig, "synthetic_attribution_histograms.png")


def memorization_examples(tables: dict[str, pd.DataFrame], family: str = "AE_Res-diff"):
    """Synthetic image next to its most similar real image, for each dataset."""
    synth, real = tables["synth"], tables["real"].set_index("image_id")
    fig, axes = plt.subplots(3, 4, figsize=(12, 7.2), gridspec_kw=dict(height_ratios=[1, 1.6, 1]))
    for r, ds in enumerate(config.DATASETS):
        d = synth[(synth.dataset == ds) & (synth.family == family) & (synth.label == "genuine")]
        picks = [d.loc[d.nn_corr.idxmax()], d.iloc[(d.nn_corr - d.nn_corr.median()).abs().argmin()]]
        for j, row in enumerate(picks):
            for k, (path, title) in enumerate([(row.path, f"synthetic {row.image_id.split('_')[-1]}"),
                                               (real.loc[row.nn_real_id].path,
                                                f"nearest real (subject {row.subject}), r={row.nn_corr:.3f}")]):
                ax = axes[r, 2 * j + k]
                ax.imshow(_img(path), cmap="gray", aspect="auto")
                ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                ax.set_title(title, fontsize=8, fontweight="normal")
            axes[r, 0].set_ylabel(ds, fontsize=11, fontweight="bold", color=INK)
    fig.suptitle(f"{family}: most similar and median-similarity synthetic genuine images vs. their nearest real image",
                 color=INK2)
    return _save(fig, f"memorization_examples_{family}.png")


# --------------------------------------------------------------------------
# Result figures
# --------------------------------------------------------------------------
def acer_heatmap(summary: pd.DataFrame, dataset: str, vmax: float | None = None):
    m = acer_matrix(summary, dataset)
    m.index = [short_combo(c) for c in m.index]
    vmax = vmax or max(5.0, float(np.nanmax(m.values)))
    fig, ax = plt.subplots(figsize=(8.2, 10))
    im = ax.imshow(m.values, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.grid(False)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.55 * vmax else INK)
    for y in (4.5, 14.5, 24.5, 29.5):                 # separators between 1/2/3/4/5-model groups
        ax.axhline(y, color=SURFACE, linewidth=2)
    ax.axvline(2.5, color=SURFACE, linewidth=2); ax.axvline(5.5, color=SURFACE, linewidth=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("pooled ACER (%)")
    ax.set_title(f"{dataset}: ACER (%) for every CNN combination and setting", pad=28)
    return _save(fig, f"heatmap_ACER_{dataset}.png")


def setting_trends(summary: pd.DataFrame):
    """ACER across settings for the five single models and the 5-model fusion (one panel per dataset)."""
    labels = [setting_label(s, f) for s, f in SETTINGS]
    series = list(config.MODEL_NAMES) + ["+".join(config.MODEL_NAMES)]
    names = [short_combo(s) if "+" not in s else "all five" for s in series]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, ds in zip(axes, config.DATASETS):
        m = acer_matrix(summary, ds).loc[series, labels]
        for i, (s, n) in enumerate(zip(series, names)):
            ax.plot(range(len(labels)), m.loc[s].values, marker="o", markersize=5, linewidth=2,
                    color=SERIES[i] if i < 5 else INK, linestyle="-" if i < 5 else "--", label=n)
        ax.set_xticks(range(len(labels)), labels, rotation=45)
        ax.set_title(ds)
        ax.set_ylabel("pooled ACER (%)")
        ax.axvline(2.5, color=GRID, linewidth=2); ax.axvline(5.5, color=GRID, linewidth=2)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return _save(fig, "trends_single_vs_all5.png")


def n_models_plot(summary: pd.DataFrame):
    """Mean ACER over all combinations with 1..5 models, per dataset and setting group."""
    groups = [("BA", "-"), ("S2", "-"), ("S4", "AE_Res-diff"), ("S4", "VAE-diff"),
              ("S5", "AE_Res-diff"), ("S5", "VAE-diff")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ds in zip(axes, config.DATASETS):
        d = summary[summary.dataset == ds]
        for i, (s, f) in enumerate(groups):
            g = d[(d.step == s) & (d.synth_family == f)].groupby("n_models").ACER.mean()
            ax.plot(g.index, g.values, marker="o", linewidth=2, markersize=5, color=SERIES[i],
                    label=setting_label(s, f))
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("number of CNNs in the combination")
        ax.set_ylabel("mean pooled ACER (%)")
        ax.set_title(ds)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return _save(fig, "acer_vs_number_of_models.png")

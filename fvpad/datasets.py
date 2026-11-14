"""
Reading the three finger-vein datasets and building the *manifest*: one table
row per image, with everything we need to know about it.

Manifest columns
----------------
dataset      IDIAP / SCUT / PLUS
image_id     unique, human-readable id of the image
path         path relative to config.DATA_ROOT
source       "real" or "synthetic"
family       "real", "AE_Res-diff" or "VAE-diff"
label        "genuine" (bona fide) or "spoof" (presentation attack)
y            0 = genuine, 1 = spoof  (the attack class is the *positive* class)
subject      subject (person) identifier, as a string
finger       finger identifier ("" if not applicable)
orig_split   the dataset's original split (train/dev/test), kept for reference
pair_key     key that links a genuine image to its spoof counterpart(s)
eligible     True if the image is used in the experiments

How subjects are identified (verified on the actual files)
-----------------------------------------------------------
IDIAP  full/{train,dev}/{real,spoof}/<SSS>-<M|F>/<SSS>_<L|R>_<session>.png
       full/test/test_file_NNN.png (anonymised; translator_full_files.txt
       gives the true name, spoofs carry the suffix "_spoof").
       subject = SSS (3 digits); finger = L/R (left/right index finger);
       110 subjects x 2 fingers x 2 sessions = 440 images per class.
SCUT   full/{train,dev,test}/{real,spoof}/<ID>/<ID>_<finger>_<session>_<shot>_<light>.bmp
       subject = client ID (= folder name); finger = 1..6.
       100 clients x 6 fingers x 6 shots = 3600 images per class.
PLUS   {real,spoof}/<UUU>/PLUS-FV3-Laser_PALMAR_<UUU>_<session>_<finger>_<image>.png
       subject = user UUU (3 digits, = folder name); finger in {02,03,04,07,08,09}.
       Only 22 of the 60 users have spoofs; each spoof file has a genuine file
       with the *identical* file name.

Genuine <-> spoof correspondence ("we only use genuine samples for which
spoof samples exist")
------------------------------------------------------------------------
IDIAP  same subject, finger side and session  -> one-to-one
SCUT   same client and finger (the SCUT documentation says only ID and finger
       matter; spoofs were captured in a different session) -> 6 vs 6 per finger
PLUS   identical file name -> one-to-one (genuine images 04/05 have no spoof)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

LABEL_TO_Y = {"genuine": 0, "spoof": 1}


def _rel(p: Path) -> str:
    return p.relative_to(config.DATA_ROOT).as_posix()


# --------------------------------------------------------------------------
# Real data
# --------------------------------------------------------------------------
_IDIAP_RE = re.compile(r"(\d{3})_([LR])_([12])(_spoof)?\.png")


def scan_idiap(root: Path) -> pd.DataFrame:
    rows = []
    full = root / "IDIAP" / "full"
    for split in ("train", "dev"):
        for cls in ("real", "spoof"):
            for f in sorted((full / split / cls).glob("*/*.png")):
                m = _IDIAP_RE.fullmatch(f.name)
                assert m and m.group(4) is None, f
                assert f.parent.name.startswith(m.group(1) + "-"), f  # folder SSS-M/F
                rows.append((f, split, cls, m.group(1), m.group(2), m.group(3)))
    translator = dict(line.split() for line in
                      (root / "IDIAP" / "translator_full_files.txt").read_text().splitlines() if line.strip())
    for f in sorted((full / "test").glob("*.png")):
        true_name = translator[f.name]
        m = _IDIAP_RE.fullmatch(true_name)
        assert m, true_name
        cls = "spoof" if m.group(4) else "real"
        rows.append((f, "test", cls, m.group(1), m.group(2), m.group(3)))

    out = []
    for f, split, cls, subj, side, sess in rows:
        label = "genuine" if cls == "real" else "spoof"
        out.append(dict(
            dataset="IDIAP", image_id=f"IDIAP_{label}_{subj}_{side}_{sess}", path=_rel(f),
            source="real", family="real", label=label, subject=subj, finger=side,
            orig_split=split, pair_key=f"{subj}_{side}_{sess}"))
    return pd.DataFrame(out)


_SCUT_RE = re.compile(r"(\d+)_(\d)_(\d)_(\d)_(\d)\.bmp")


def scan_scut(root: Path) -> pd.DataFrame:
    out = []
    for f in sorted((root / "SCUT" / "full").glob("*/*/*/*.bmp")):
        split, cls, folder = f.parts[-4], f.parts[-3], f.parts[-2]
        m = _SCUT_RE.fullmatch(f.name)
        assert m and m.group(1) == folder, f
        cid, finger, sess, shot, light = m.groups()
        label = "genuine" if cls == "real" else "spoof"
        out.append(dict(
            dataset="SCUT", image_id=f"SCUT_{label}_{cid}_{finger}_{sess}_{shot}_{light}",
            path=_rel(f), source="real", family="real", label=label,
            subject=f"{int(cid):03d}", finger=finger, orig_split=split,
            pair_key=f"{int(cid):03d}_{finger}", shot=int(shot)))
    return pd.DataFrame(out)


_PLUS_RE = re.compile(r"(PLUS-FV3-(?:Laser|LED))_(PALMAR|DORSAL)_(\d{3})_(\d{2})_(\d{2})_(\d{2})\.png")


def scan_plus(root: Path) -> pd.DataFrame:
    out = []
    for cls in ("real", "spoof"):
        for f in sorted((root / "PLUS" / cls).glob("*/*.png")):
            m = _PLUS_RE.fullmatch(f.name)
            assert m and m.group(3) == f.parent.name, f
            scanner, side, user, sess, finger, img = m.groups()
            label = "genuine" if cls == "real" else "spoof"
            out.append(dict(
                dataset="PLUS", image_id=f"PLUS_{label}_{user}_{sess}_{finger}_{img}",
                path=_rel(f), source="real", family="real", label=label, subject=user,
                finger=finger, orig_split="", pair_key=f.stem))
    return pd.DataFrame(out)


def mark_eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Keep genuine images that have a spoof counterpart (and vice versa).

    Within each pair_key we keep min(#genuine, #spoof) images of each class,
    so every key - and therefore every subject and fold - is exactly balanced.
    For IDIAP and PLUS the keys are one-to-one; for SCUT each key (client,
    finger) has 6 genuine and 6 spoof images.
    """
    df = df.copy()
    df["eligible"] = False
    sort_cols = ["image_id"]
    for key, g in df.groupby("pair_key"):
        gen = g[g.label == "genuine"].sort_values(sort_cols)
        spo = g[g.label == "spoof"].sort_values(sort_cols)
        n = min(len(gen), len(spo))
        df.loc[gen.index[:n], "eligible"] = True
        df.loc[spo.index[:n], "eligible"] = True
    return df


def build_real_manifest(root: Path = config.REAL_ROOT) -> pd.DataFrame:
    frames = [mark_eligible(scan_idiap(root)), mark_eligible(scan_scut(root)), mark_eligible(scan_plus(root))]
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df["label"].map(LABEL_TO_Y)
    assert df.image_id.is_unique
    return df


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
SYNTH_FOLDER = {"IDIAP": "IDIAP", "SCUT": "SCUT", "PLUS": "PLUS_matched"}


def scan_synthetic(family: str, root: Path) -> pd.DataFrame:
    """List synthetic images. Files are named sample_<n>.png - they carry no
    subject, finger or source-image information (verified: no PNG metadata
    either). Subjects are attributed later by `attribute_synthetic`."""
    out = []
    for ds, folder in SYNTH_FOLDER.items():
        for cls, label in (("real", "genuine"), ("spoof", "spoof")):
            files = sorted((root / folder / cls / "samples").glob("sample_*.png"),
                           key=lambda p: int(p.stem.split("_")[1]))
            for f in files:
                out.append(dict(
                    dataset=ds, image_id=f"{ds}_{family}_{label}_{f.stem}", path=_rel(f),
                    source="synthetic", family=family, label=label, subject="", finger="",
                    orig_split="", pair_key="", eligible=True))
    df = pd.DataFrame(out)
    df["y"] = df["label"].map(LABEL_TO_Y)
    return df


# Down-scaling factor for the similarity search. 1/4 of the native size keeps
# enough vein/texture detail to recognise near-copies while staying fast.
ATTRIBUTION_SCALE = 4


def _pixel_vectors(paths: list[str], size: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    vecs = np.empty((len(paths), size[0] * size[1]), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(config.DATA_ROOT / p) as im:
            g = np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.float32).ravel()
        vecs[i] = (g - g.mean()) / (g.std() + 1e-6)          # zero mean, unit variance
    return vecs


def attribute_synthetic(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link every synthetic image to its most similar *eligible real* image of
    the same dataset and class, using the Pearson correlation of down-scaled
    grayscale images.

    Adds the columns
      nn_real_id    image_id of the most similar real image
      nn_corr       its correlation
      nn_corr_other best correlation with a real image of a *different* subject
      subject       subject of the most similar real image (attributed subject)
      attributed    True if nn_corr is higher than the highest correlation ever
                    observed between real images of two different subjects of
                    that dataset/class (threshold `tau`). Such an image is closer
                    to one particular real image than any two different people's
                    images are to each other, i.e. it is a near-copy of it.

    Returns (synth_with_columns, thresholds_table).
    """
    synth = synth.copy()
    for c in ("nn_real_id", "subject"):
        synth[c] = ""
    synth["nn_corr"] = np.nan
    synth["nn_corr_other"] = np.nan
    synth["attributed"] = False
    thr_rows = []
    for ds in config.DATASETS:
        native = _native_size(real[(real.dataset == ds)].path.iloc[0])
        size = (native[0] // ATTRIBUTION_SCALE, native[1] // ATTRIBUTION_SCALE)
        for label in ("genuine", "spoof"):
            r = real[(real.dataset == ds) & (real.label == label) & real.eligible].reset_index(drop=True)
            R = _pixel_vectors(r.path.tolist(), size)
            subj = r.subject.to_numpy()
            # threshold: max similarity between real images of different subjects
            C = R @ R.T / R.shape[1]
            C[subj[:, None] == subj[None, :]] = -np.inf
            tau = float(C.max())
            thr_rows.append(dict(dataset=ds, label=label, tau=tau,
                                 median_real_nn_other_subject=float(np.median(C.max(1)))))
            for fam in synth.family.unique():
                idx = synth.index[(synth.dataset == ds) & (synth.label == label) & (synth.family == fam)]
                if len(idx) == 0:
                    continue
                S = _pixel_vectors(synth.loc[idx, "path"].tolist(), size)
                C = S @ R.T / R.shape[1]
                j = C.argmax(1)
                best = C[np.arange(len(j)), j]
                C[subj[None, :] == subj[j][:, None]] = -np.inf
                other = C.max(1)
                synth.loc[idx, "nn_real_id"] = r.image_id.to_numpy()[j]
                synth.loc[idx, "subject"] = subj[j]
                synth.loc[idx, "nn_corr"] = best
                synth.loc[idx, "nn_corr_other"] = other
                synth.loc[idx, "attributed"] = best > tau
    return synth, pd.DataFrame(thr_rows)


def _native_size(rel_path: str) -> tuple[int, int]:
    from PIL import Image
    with Image.open(config.DATA_ROOT / rel_path) as im:
        return im.size  # (width, height)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def summarize_real(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset", sort=False):
        e = g[g.eligible]
        rows.append(dict(
            dataset=ds,
            genuine_images=int((g.label == "genuine").sum()), spoof_images=int((g.label == "spoof").sum()),
            subjects_total=g.subject.nunique(),
            subjects_with_spoofs=g[g.label == "spoof"].subject.nunique(),
            eligible_genuine=int((e.label == "genuine").sum()), eligible_spoof=int((e.label == "spoof").sum()),
            eligible_subjects=e.subject.nunique()))
    return pd.DataFrame(rows)

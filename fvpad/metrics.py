"""
PAD error rates (ISO/IEC 30107-3), as defined in the teacher's SAC paper.

Class convention used everywhere in this project
    y = 1  spoof / presentation attack  -> the POSITIVE class
    y = 0  genuine / bona fide          -> the negative class

    TP  attack classified as attack          FN  attack classified as bona fide
    TN  bona fide classified as bona fide     FP  bona fide classified as attack

    APCER = FN / (TP + FN)   share of attacks that were accepted as bona fide
    BPCER = FP / (FP + TN)   share of bona fide samples rejected as attacks
    ACER  = (APCER + BPCER) / 2

All three are reported in percent; lower is better.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return dict(TP=int(((y_true == 1) & (y_pred == 1)).sum()),
                FN=int(((y_true == 1) & (y_pred == 0)).sum()),
                FP=int(((y_true == 0) & (y_pred == 1)).sum()),
                TN=int(((y_true == 0) & (y_pred == 0)).sum()))


def rates(TP: int, FN: int, FP: int, TN: int) -> dict[str, float]:
    apcer = 100.0 * FN / (TP + FN) if (TP + FN) else np.nan
    bpcer = 100.0 * FP / (FP + TN) if (FP + TN) else np.nan
    return dict(APCER=apcer, BPCER=bpcer, ACER=(apcer + bpcer) / 2.0)


def pad_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    return {**c, **rates(**c)}

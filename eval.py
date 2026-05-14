"""
eval.py
=======
Evaluation metrics for MAG-based CME detection.

Standard ML metrics (F1, precision, recall) plus space weather
specific metrics used by forecasters:

  POD  : Probability of Detection = TP / (TP + FN)
         "What fraction of real CMEs did we catch?"

  FAR  : False Alarm Rate = FP / (FP + TN)
         Space weather definition -- NOT the ML definition.
         "What fraction of quiet periods did we falsely flag?"

  CSI  : Critical Success Index = TP / (TP + FP + FN)
         Also called Threat Score. Ignores true negatives entirely.
         Standard benchmark metric in operational space weather.

  HSS  : Heidke Skill Score -- measures skill above random chance.
         HSS = 0: no skill (same as random).
         HSS = 1: perfect. HSS < 0: worse than random.
         Most meaningful single metric for rare-event forecasting.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)


def evaluate(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Full evaluation suite for binary CME detection.

    Parameters
    ----------
    y_true     : ground truth labels (0 or 1)
    y_prob     : predicted probabilities in [0, 1]
    threshold  : decision boundary (default 0.5)

    Returns
    -------
    dict of metric name -> float value
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Standard ML metrics
    f1        = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)

    # Space weather metrics
    pod = tp / (tp + fn + 1e-9)                   # same as recall
    far = fp / (fp + tn + 1e-9)                   # space weather FAR
    csi = tp / (tp + fp + fn + 1e-9)

    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    hss     = hss_num / (hss_den + 1e-9)

    # AUC metrics (threshold-independent)
    n_pos = int(y_true.sum())
    if n_pos > 0 and n_pos < len(y_true):
        auc_roc = float(roc_auc_score(y_true, y_prob))
        auc_pr  = float(average_precision_score(y_true, y_prob))
    else:
        auc_roc = float("nan")
        auc_pr  = float("nan")

    return {
        "f1":        float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "pod":       float(pod),
        "far":       float(far),
        "csi":       float(csi),
        "hss":       float(hss),
        "auc_roc":   auc_roc,
        "auc_pr":    auc_pr,
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "threshold": threshold,
    }


def print_report(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print()
    print("=" * 50)
    print("  CME Detection Evaluation Report")
    print("=" * 50)
    print(f"  Threshold  : {metrics['threshold']:.2f}")
    print()
    print("  -- Standard ML --")
    print(f"  F1 Score   : {metrics['f1']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  AUC-ROC    : {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR     : {metrics['auc_pr']:.4f}")
    print()
    print("  -- Space Weather --")
    print(f"  POD        : {metrics['pod']:.4f}  (probability of detection)")
    print(f"  FAR        : {metrics['far']:.4f}  (false alarm rate)")
    print(f"  CSI        : {metrics['csi']:.4f}  (critical success index)")
    print(f"  HSS        : {metrics['hss']:.4f}  (Heidke skill score)")
    print()
    print("  -- Confusion Matrix --")
    print(f"  TP={metrics['tp']:4d}  FP={metrics['fp']:4d}")
    print(f"  FN={metrics['fn']:4d}  TN={metrics['tn']:4d}")
    print("=" * 50)
    print()


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    steps:  int = 50,
) -> tuple[float, dict]:
    """
    Sweep thresholds from 0.1 to 0.9 and return the one that
    maximises the chosen metric.

    Parameters
    ----------
    metric : one of "f1", "csi", "hss", "pod"

    Returns
    -------
    (best_threshold, metrics_at_best_threshold)
    """
    best_thresh  = 0.5
    best_score   = -float("inf")
    best_metrics = {}

    for thresh in np.linspace(0.1, 0.9, steps):
        m = evaluate(y_true, y_prob, threshold=thresh)
        if m[metric] > best_score:
            best_score   = m[metric]
            best_thresh  = thresh
            best_metrics = m

    return float(best_thresh), best_metrics
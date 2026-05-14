# eval.py
"""
CME detection evaluation metrics.
Beyond F1: FAR, probability of detection (POD), lead time bias.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Full evaluation suite for CME binary detection.

    Space weather metrics:
      POD  = Probability of Detection = recall (hit rate)
      FAR  = False Alarm Rate = FP / (FP + TN)  [space weather definition]
      CSI  = Critical Success Index = TP / (TP + FP + FN)
      HSS  = Heidke Skill Score — measures skill vs random chance
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pod  = tp / (tp + fn + 1e-9)                        # recall
    far  = fp / (fp + tn + 1e-9)                        # space weather FAR
    csi  = tp / (tp + fp + fn + 1e-9)
    hss_num = 2 * (tp * tn - fp * fn)
    hss_den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss  = hss_num / (hss_den + 1e-9)

    return {
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_pod": pod,
        "far":       far,
        "csi":       csi,
        "hss":       hss,
        "auc_roc":   roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def print_report(metrics: dict):
    print("\n── CME Detection Evaluation ──────────────────")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  POD       : {metrics['recall_pod']:.4f}  (probability of detection)")
    print(f"  FAR       : {metrics['far']:.4f}  (false alarm rate)")
    print(f"  CSI       : {metrics['csi']:.4f}  (critical success index)")
    print(f"  HSS       : {metrics['hss']:.4f}  (Heidke skill score)")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"  Confusion : TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print("──────────────────────────────────────────────\n")
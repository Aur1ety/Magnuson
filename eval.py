"""
eval.py
=======
CME detection evaluation metrics.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, precision_score, roc_auc_score, confusion_matrix

def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pod  = tp / (tp + fn + 1e-9)                        
    far  = fp / (fp + tn + 1e-9)                        
    csi  = tp / (tp + fp + fn + 1e-9)
    hss  = (2 * (tp * tn - fp * fn)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-9)

    return {
        "f1": f1_score(y_true, y_pred, zero_division=0), "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_pod": pod, "far": far, "csi": csi, "hss": hss,
        "auc_roc": roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else np.nan,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

def print_report(metrics: dict):
    print("\n── CME Detection Evaluation ──────────────────")
    for k, v in metrics.items(): print(f"  {k.upper():<10} : {v:.4f}" if isinstance(v, float) else f"  {k.upper():<10} : {v}")
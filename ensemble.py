"""
ensemble.py
===========
Weighted ensemble combining TCN and BzLSTM predictions.

Default weights:
  TCN    0.50 -- stronger at sharp transients (shock, sudden Bz drop)
  BzLSTM 0.50 -- stronger at sustained rotation (flux rope passage)

Equal weights as starting point -- tune using validation F1 per model
on your holdout set. If TCN F1 >> BzLSTM F1, increase TCN weight.

Usage
-----
    from ensemble import CMEEnsemble

    ens = CMEEnsemble(
        tcn_path    = "saved_models/tcn_mag_v1.pth",
        bzlstm_path = "saved_models/bzlstm_mag_v1.pth",
    )
    result = ens.predict_latest(X_window)
    print(result)
    # {'p_cme': 0.82, 'verdict': 'CME DETECTED', 'p_tcn': 0.79, 'p_bzlstm': 0.85}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model_factory import TCNModel, BzLSTM


class CMEEnsemble:
    """
    Weighted ensemble of TCN + BzLSTM for MAG-based CME detection.

    Parameters
    ----------
    tcn_path    : path to saved TCN state dict (.pth)
    bzlstm_path : path to saved BzLSTM state dict (.pth)
    tcn_weight  : ensemble weight for TCN (default 0.5)
    device      : "cuda" or "cpu" (auto-detected if not specified)
    input_dim   : number of MAG features (8)
    """

    def __init__(
        self,
        tcn_path:    str,
        bzlstm_path: str,
        tcn_weight:  float = 0.5,
        device:      str   = "auto",
        input_dim:   int   = 8,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.tcn_weight = tcn_weight
        self.bzlstm_weight = 1.0 - tcn_weight

        self.tcn    = self._load(TCNModel(input_dim=input_dim), tcn_path)
        self.bzlstm = self._load(BzLSTM(input_dim=input_dim),  bzlstm_path)

    def _load(self, model: torch.nn.Module, path: str) -> torch.nn.Module:
        if Path(path).exists():
            model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}\n"
                "Run train.py first to generate saved_models/"
            )
        return model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run ensemble inference on a batch of sequences.

        Parameters
        ----------
        x : np.ndarray of shape (N, seq_len, 8)

        Returns
        -------
        dict with keys:
            p_tcn      : TCN probabilities (N,)
            p_bzlstm   : BzLSTM probabilities (N,)
            p_ensemble : weighted average (N,)
        """
        t = torch.tensor(x, dtype=torch.float32, device=self.device)

        p_tcn    = torch.sigmoid(self.tcn(t)).cpu().numpy()
        p_bzlstm = torch.sigmoid(self.bzlstm(t)).cpu().numpy()
        p_ens    = self.tcn_weight * p_tcn + self.bzlstm_weight * p_bzlstm

        return {
            "p_tcn":      p_tcn,
            "p_bzlstm":   p_bzlstm,
            "p_ensemble": p_ens,
        }

    def predict_latest(
        self,
        x:         np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """
        Run inference on the most recent window only.

        Parameters
        ----------
        x         : np.ndarray of shape (N, seq_len, 8) or (seq_len, 8)
        threshold : decision boundary

        Returns
        -------
        dict with probability scores and verdict string
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]   # add batch dim

        results = self.predict(x[-1:])
        score   = float(results["p_ensemble"][0])

        return {
            "p_cme":      score,
            "p_tcn":      float(results["p_tcn"][0]),
            "p_bzlstm":   float(results["p_bzlstm"][0]),
            "verdict":    "CME DETECTED" if score >= threshold else "QUIET",
            "threshold":  threshold,
        }

    def update_weights(
        self,
        y_true:    np.ndarray,
        y_prob_tcn: np.ndarray,
        y_prob_bzlstm: np.ndarray,
    ) -> None:
        """
        Auto-tune ensemble weights based on validation F1 scores.
        Assigns higher weight to whichever model has higher F1.

        Call this after evaluating both models on your validation set.
        """
        from sklearn.metrics import f1_score

        f1_tcn    = f1_score(y_true, (y_prob_tcn    >= 0.5).astype(int), zero_division=0)
        f1_bzlstm = f1_score(y_true, (y_prob_bzlstm >= 0.5).astype(int), zero_division=0)

        total = f1_tcn + f1_bzlstm + 1e-9
        self.tcn_weight    = f1_tcn    / total
        self.bzlstm_weight = f1_bzlstm / total

        print(
            f"Weights updated -- TCN: {self.tcn_weight:.3f} | "
            f"BzLSTM: {self.bzlstm_weight:.3f} "
            f"(F1: {f1_tcn:.3f} vs {f1_bzlstm:.3f})"
        )
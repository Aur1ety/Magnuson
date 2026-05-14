# ensemble.py
"""
Weighted ensemble of TCN, TCAN, and BzLSTM.
"""
from __future__ import annotations

import torch
import numpy as np
from pathlib import Path


class CMEEnsemble:
    """
    Weights (default):
      TCN    0.50 — best at plasma shock detection (fast transients)
      TCAN   0.30 — long-range precursor via self-attention
      BzLSTM 0.20 — flux rope Bz rotation over hours

    Tune weights using validation F1 per model on your holdout set.
    """

    WEIGHTS = {"tcn": 0.50, "tcan": 0.30, "bzlstm": 0.20}

    def __init__(
        self,
        tcn_path:    str,
        tcan_path:   str,
        bzlstm_path: str,
        device: str = "cpu",
    ):
        from model_factory import TCNModel, TCANModel, BzLSTM  # your existing imports

        self.device = torch.device(device)

        self.tcn    = self._load(TCNModel(input_dim=14),    tcn_path)
        self.tcan   = self._load(TCANModel(input_dim=14),   tcan_path)
        self.bzlstm = self._load(BzLSTM(input_dim=14),      bzlstm_path)

    def _load(self, model, path):
        if Path(path).exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        x : np.ndarray of shape (N, seq_len, 14)

        Returns
        -------
        np.ndarray of shape (N,) — P(CME) in [0, 1]
        """
        t = torch.tensor(x, dtype=torch.float32, device=self.device)

        p_tcn    = torch.sigmoid(self.tcn(t)).cpu().numpy()
        p_tcan   = torch.sigmoid(self.tcan(t)).cpu().numpy()
        p_bzlstm = torch.sigmoid(self.bzlstm(t)).cpu().numpy()

        w = self.WEIGHTS
        p_ensemble = (
            w["tcn"]    * p_tcn  +
            w["tcan"]   * p_tcan +
            w["bzlstm"] * p_bzlstm
        )
        return p_ensemble

    def predict_latest(self, x: np.ndarray, threshold: float = 0.5) -> dict:
        """Convenience wrapper for inference on the most recent window."""
        p = self.predict(x[-1:])
        score = float(p[0])
        return {
            "p_cme":    score,
            "verdict":  "CME DETECTED" if score >= threshold else "QUIET",
            "threshold": threshold,
        }
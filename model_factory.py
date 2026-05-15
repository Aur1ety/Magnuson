"""
model_factory.py
================
All model architectures for MAG-based CME detection benchmarking.

Models:
  1. TCNModel              - Temporal Convolutional Network (baseline)
  2. TransformerModel      - Self-attention encoder
  3. TFTModel              - Temporal Fusion Transformer (variable importance)
  4. CNNTransformer        - 1D CNN local features + Transformer global context
  5. XGBoostModel          - Gradient boosted trees on summary statistics
  6. PatchTransformer      - ViT-style patch attention (NEW)
  7. LightGBMModel         - LightGBM + SHAP interpretability (NEW)
  8. EnsembleTFTTransTCN   - Learned-weight ensemble of TFT+Transformer+TCN (NEW)

All deep learning models:
  Input:  (batch, seq_len, 8)  -- 8 MAG features
  Output: (batch,)             -- raw logit (apply sigmoid for probability)

XGBoost / LightGBM:
  Input:  (N, seq_len, 8)     -- flattened to summary stats internally
  Output: (N,)                -- probability in [0, 1] directly
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── 1. TCN ────────────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(in_ch, out_ch, kernel_size,
                                 dilation=dilation, padding=self.padding)
        self.norm    = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()

    def forward(self, x):
        out = self.conv(x)[:, :, :x.size(2)]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.act(out))


class ResidualTCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch,  out_ch, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation, dropout)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act   = nn.GELU()

    def forward(self, x):
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network.
    Best at sharp transients: shock arrival, sudden Bz southward turning.
    Receptive field ~125 steps (~2hrs) with 4 blocks at kernel=3.
    """
    def __init__(self, input_dim=8, n_filters=32, kernel_size=3,
                 n_blocks=4, dropout=0.4):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, n_filters, 1)
        self.blocks     = nn.Sequential(*[
            ResidualTCNBlock(n_filters, n_filters, kernel_size, 2**i, dropout)
            for i in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(n_filters, n_filters // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(n_filters // 2, 1),
        )

    def forward(self, x):
        x = self.blocks(self.input_proj(x.transpose(1, 2)))
        return self.head(x).squeeze(1)


# ── 2. Transformer Encoder ────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    """
    Transformer Encoder for CME flux rope detection.
    Self-attention directly connects any two time steps regardless of distance.
    Multi-head attention can simultaneously track Bz rotation AND |B| enhancement.
    """
    def __init__(self, input_dim=8, d_model=64, nhead=4,
                 num_layers=3, dropout=0.2, max_seq_len=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=max_seq_len,
                                             dropout=dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        x = self.pos_enc(self.input_proj(x))
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# ── 3. Temporal Fusion Transformer ───────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.gate     = nn.Linear(hidden_dim, output_dim)
        self.skip     = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm     = nn.LayerNorm(output_dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        h    = F.elu(self.fc1(x))
        h    = self.dropout(h)
        out  = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(gate * out + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.grns      = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(input_dim)
        ])
        self.softmax   = nn.Linear(input_dim * hidden_dim, input_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        b, t, f = x.shape
        processed = []
        for i, grn in enumerate(self.grns):
            feat = x[:, :, i:i+1]
            processed.append(grn(feat.reshape(b*t, 1)).reshape(b, t, -1))
        stacked   = torch.stack(processed, dim=-1)
        flat      = torch.cat(processed, dim=-1)
        weights   = torch.softmax(self.softmax(flat), dim=-1).unsqueeze(2)
        combined  = (stacked * weights).sum(dim=-1)
        return combined, weights.squeeze(2)


class TFTModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=32, nhead=4,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        x, _   = self.vsn(x)
        x      = self.transformer(x)
        x      = x.mean(dim=1)
        return self.head(x).squeeze(-1)

    def get_feature_importance(self, x):
        _, weights = self.vsn(x)
        return weights.mean(dim=(0, 1))


# ── 4. CNN + Transformer Hybrid ──────────────────────────────────────────────

class CNNTransformer(nn.Module):
    def __init__(self, input_dim=8, cnn_filters=32, d_model=64,
                 nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(cnn_filters, cnn_filters, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj    = nn.Linear(cnn_filters, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        cnn_out = self.cnn(x.transpose(1, 2))
        cnn_out = cnn_out.transpose(1, 2)
        x       = self.pos_enc(self.proj(cnn_out))
        x       = self.transformer(x)
        x       = x.mean(dim=1)
        return self.head(x).squeeze(-1)


# ── 5. XGBoost Wrapper ───────────────────────────────────────────────────────

class XGBoostModel:
    def __init__(self, n_estimators=300, max_depth=6,
                 learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, scale_pos_weight=20):
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="aucpr",
                random_state=42,
                n_jobs=-1,
            )
            self.fitted = False
        except ImportError:
            raise ImportError("pip install xgboost")

    @staticmethod
    def extract_features(X: np.ndarray) -> np.ndarray:
        from scipy.stats import skew, kurtosis
        N, T, F = X.shape
        stats = []
        for i in range(N):
            window = X[i]
            row = []
            for f in range(F):
                col = window[:, f]
                row.extend([
                    col.mean(), col.std(), col.min(), col.max(),
                    col.max() - col.min(), col[-1], col[0],
                    np.polyfit(np.arange(T), col, 1)[0],
                    float(skew(col)), float(kurtosis(col)),
                ])
            bz_col   = window[:, 0]
            pers_col = window[:, 6]
            rot_col  = window[:, 4]
            row.extend([
                bz_col.min(), pers_col.max(), rot_col.sum(),
                (bz_col < -10).sum(), (bz_col < -20).sum(),
            ])
            stats.append(row)
        return np.array(stats, dtype=np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        Xf = self.extract_features(X_train)
        eval_set = None
        if X_val is not None:
            eval_set = [(self.extract_features(X_val), y_val)]
        self.model.fit(Xf, y_train, eval_set=eval_set, verbose=False)
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        Xf = self.extract_features(X)
        return self.model.predict_proba(Xf)[:, 1]

    def feature_importance(self, feature_names=None):
        import pandas as pd
        imp = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": feature_names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(15)


# ── 6. Patch Transformer (NEW) ────────────────────────────────────────────────

class PatchTransformer(nn.Module):
    """
    ViT-style patch attention for time series (PatchTST approach).
    """
    def __init__(self, input_dim=8, patch_size=16, d_model=64, nhead=4,
                 num_layers=3, dropout=0.2, seq_len=128):
        super().__init__()
        assert seq_len % patch_size == 0

        self.patch_size  = patch_size
        self.num_patches = seq_len // patch_size

        self.patch_embed = nn.Linear(patch_size * input_dim, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        B, T, F = x.shape
        x = x.reshape(B, self.num_patches, self.patch_size * F)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        return self.head(cls_out).squeeze(-1)


# ── 7. LightGBM + SHAP (NEW) ─────────────────────────────────────────────────

class LightGBMModel:
    def __init__(self, n_estimators=500, max_depth=6,
                 learning_rate=0.05, num_leaves=31,
                 subsample=0.8, colsample_bytree=0.8,
                 scale_pos_weight=20, min_child_samples=5):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                min_child_samples=min_child_samples,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            self.fitted       = False
            self._explainer   = None
            self._feat_names  = None
        except ImportError:
            raise ImportError("pip install lightgbm")

    @staticmethod
    def extract_features(X: np.ndarray) -> np.ndarray:
        return XGBoostModel.extract_features(X)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            feature_names=None):
        Xf = self.extract_features(X_train)
        self._feat_names = feature_names

        callbacks = []
        eval_set  = None
        if X_val is not None:
            try:
                import lightgbm as lgb
                callbacks = [lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)]
            except Exception:
                pass
            eval_set = [(self.extract_features(X_val), y_val)]

        self.model.fit(
            Xf, y_train,
            eval_set=eval_set,
            callbacks=callbacks if eval_set else None,
        )
        self.fitted = True

        try:
            import shap
            self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            self._explainer = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self.model.predict_proba(self.extract_features(X))[:, 1]

    def feature_importance(self, feature_names=None):
        import pandas as pd
        imp   = self.model.feature_importances_
        names = feature_names or self._feat_names or [f"feat_{i}" for i in range(len(imp))]
        df    = pd.DataFrame({"feature": names, "importance": imp})
        return df.sort_values("importance", ascending=False).head(15)

    def shap_summary(self, X: np.ndarray, feature_names=None) -> "pd.DataFrame":
        import pandas as pd
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not available.")
        Xf         = self.extract_features(X)
        shap_vals  = self._explainer.shap_values(Xf)
        sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        names = feature_names or self._feat_names or [f"feat_{i}" for i in range(Xf.shape[1])]
        df = pd.DataFrame({
            "feature":     names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        })
        return df.sort_values("mean_abs_shap", ascending=False).head(20)

    def shap_explain(self, x_single: np.ndarray, feature_names=None) -> "pd.DataFrame":
        import pandas as pd
        if self._explainer is None:
            raise RuntimeError("SHAP explainer not available.")
        Xf        = self.extract_features(x_single[np.newaxis])
        shap_vals = self._explainer.shap_values(Xf)
        sv        = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        names     = feature_names or self._feat_names or [f"feat_{i}" for i in range(len(sv))]
        df        = pd.DataFrame({"feature": names, "shap_value": sv})
        return df.reindex(df["shap_value"].abs().sort_values(ascending=False).index).head(15)


# ── 8. Ensemble: TFT + Transformer + TCN (NEW) ───────────────────────────────

class EnsembleTFTTransTCN(nn.Module):
    def __init__(self, tft: TFTModel, transformer: TransformerModel,
                 tcn: TCNModel, frozen: bool = True):
        super().__init__()
        self.tft         = tft
        self.transformer = transformer
        self.tcn         = tcn

        if frozen:
            for m in [self.tft, self.transformer, self.tcn]:
                for p in m.parameters():
                    p.requires_grad_(False)

        self.blend = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        l_tft   = self.tft(x)
        l_trans = self.transformer(x)
        l_tcn   = self.tcn(x)
        stacked = torch.stack([l_tft, l_trans, l_tcn], dim=1)
        return self.blend(stacked).squeeze(-1)

    @classmethod
    def from_saved(cls, tft_path: str, transformer_path: str, tcn_path: str,
                   input_dim: int = 8, frozen: bool = True,
                   device: str | None = None) -> "EnsembleTFTTransTCN":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        def _load(model_cls, path, **kwargs):
            m = model_cls(input_dim=input_dim, **kwargs)
            m.load_state_dict(torch.load(path, map_location=device))
            return m

        tft         = _load(TFTModel,         tft_path)
        transformer = _load(TransformerModel, transformer_path)
        tcn         = _load(TCNModel,         tcn_path)

        return cls(tft, transformer, tcn, frozen=frozen)


# ── Stubs for backwards compatibility ────────────────────────────────────────
class BzLSTM(TransformerModel):
    pass

class TCANModel(TCNModel):
    pass
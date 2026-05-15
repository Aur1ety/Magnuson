"""
model_factory.py
================
TCN and BzLSTM model architectures for MAG-based CME detection.

TCNModel  : Temporal Convolutional Network
            Best at sharp transient patterns -- CME shock arrival,
            sudden field compression, abrupt Bz southward turning.

BzLSTM    : LSTM with learned feature attention
            Best at slow sustained patterns -- Bz rotation over 6-12 hrs,
            persistent southward field, smooth magnetic cloud passage.

Both models:
  - Input:  (batch, seq_len, 8)  -- 8 MAG features
  - Output: (batch,)             -- raw logit (apply sigmoid for probability)
  - Trained with focal loss to handle class imbalance (~5% CME rate)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── TCN building blocks ───────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """
    Causal dilated 1D convolution -- no future leakage.
    Pads left only so output at time t only sees t and earlier.
    LayerNorm + GELU + Dropout applied after convolution.
    """
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int   = 3,
        dilation:     int   = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv    = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.norm    = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act     = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        out = self.conv(x)[:, :, : x.size(2)]          # trim causal padding
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.act(out))


class ResidualTCNBlock(nn.Module):
    """Two causal convolutions with a residual (skip) connection."""
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int   = 3,
        dilation:     int   = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels,  out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)
        self.skip  = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


class TCNModel(nn.Module):
    """
    Temporal Convolutional Network for CME shock detection.

    Architecture:
      Input projection → 6 residual TCN blocks with exponentially
      increasing dilation (1, 2, 4, 8, 16, 32) → global average pool
      → 2-layer MLP classifier.

    Dilation doubles each block so the receptive field grows
    exponentially: with kernel=3 and 6 blocks the model sees
      2 * (3-1) * (1+2+4+8+16+32) - 1 = 125 time steps back.
    At 1-min cadence that is ~2 hours of context per prediction.

    Parameters
    ----------
    input_dim : number of input features (8 for MAG-only)
    n_filters : convolutional channel width
    kernel_size : convolution kernel size
    n_blocks  : number of residual blocks (controls receptive field)
    dropout   : dropout rate applied after each conv layer
    """
    def __init__(
        self,
        input_dim:   int   = 8,
        n_filters:   int   = 32,
        kernel_size: int   = 3,
        n_blocks:    int   = 4,
        dropout:     float = 0.4,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, n_filters, kernel_size=1)
        self.tcn_blocks = nn.Sequential(*[
            ResidualTCNBlock(n_filters, n_filters, kernel_size, 2**i, dropout)
            for i in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_filters, n_filters // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_filters // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> transpose for Conv1d
        x = self.input_proj(x.transpose(1, 2))    # (batch, n_filters, seq_len)
        x = self.tcn_blocks(x)
        return self.head(x).squeeze(1)             # (batch,)


# ── BzLSTM ────────────────────────────────────────────────────────────────────

class FeatureAttention(nn.Module):
    """
    Learned per-feature scaling applied before the LSTM.
    Allows the model to up-weight the Bz-family features
    (bz, bz_smoothed, bz_persistence) automatically during training
    rather than requiring manual hardcoding.

    Initialised to ones (no-op) and trained via backprop.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        return x * torch.softmax(self.weights, dim=0) * len(self.weights)


class BzLSTM(nn.Module):
    """
    LSTM with feature attention, specialised for Bz flux rope detection.

    Why LSTM over TCN for this role?
    The TCN has a fixed receptive field (~2 hrs with 6 blocks).
    CME flux rope Bz rotation takes 6-12 hours -- an LSTM with
    sequence length 128 at 1-min cadence sees ~2 hrs, but its hidden
    state carries context across the full sequence, effectively giving
    it unlimited memory for the slow rotation signature.

    Architecture:
      Feature attention → linear projection → 2-layer LSTM
      → last hidden state → 2-layer MLP classifier

    Parameters
    ----------
    input_dim  : number of input features (8 for MAG-only)
    hidden_dim : LSTM hidden state size
    num_layers : number of stacked LSTM layers
    dropout    : dropout between LSTM layers and in classifier
    """
    def __init__(
        self,
        input_dim:  int   = 8,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.attention   = FeatureAttention(input_dim)
        self.input_proj  = nn.Linear(input_dim, input_dim)
        self.lstm        = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = self.input_proj(self.attention(x))
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, batch, hidden)
        last_hidden  = h_n[-1]              # top layer: (batch, hidden)
        return self.classifier(last_hidden).squeeze(-1)   # (batch,)


# ── Stub for backwards compatibility ─────────────────────────────────────────
# ensemble.py imports TCANModel -- this prevents ImportError
# if you decide to add TCAN later, replace this stub.
class TCANModel(TCNModel):
    """Stub -- inherits TCNModel. Replace with full TCAN if needed."""
    pass
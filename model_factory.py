"""
model_factory.py
================
TCN and Attention-Weighted BzLSTM Architectures
"""
from __future__ import annotations
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)[:, :, : x.size(2)]
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.act(out))

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))

class TCNModel(nn.Module):
    def __init__(self, input_dim: int = 14, n_filters: int = 64, kernel_size: int = 3, n_blocks: int = 6, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, n_filters, kernel_size=1)
        self.tcn_blocks = nn.Sequential(*[ResidualTCNBlock(n_filters, n_filters, kernel_size, 2**i, dropout) for i in range(n_blocks)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(n_filters, n_filters // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(n_filters // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn_blocks(self.input_proj(x.transpose(1, 2)))
        return self.head(x).squeeze(1)

class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights

class BzLSTM(nn.Module):
    def __init__(self, input_dim: int = 14, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.attention = FeatureAttention(input_dim)
        self.input_proj = nn.Linear(input_dim, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout/2), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(self.input_proj(self.attention(x)))
        return self.classifier(lstm_out[:, -1, :]).squeeze(-1)

# Include TCANModel stub to prevent import errors in train.py / ensemble.py if you haven't written it yet
class TCANModel(TCNModel):
    pass
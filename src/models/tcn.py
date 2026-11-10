"""Temporal Convolutional Network classifier baseline (Bai et al., 2018).

Non-recurrent (convolutional) counterpart to the BiLSTM/Transformer, used to test
architecture-robustness of the observability conclusions. Linear(d->C) +
LayerNorm -> stacked dilated residual Conv1d blocks (kernel 3, dilation 2^i,
weight-norm, GELU, dropout) -> LayerNorm -> masked mean-pool -> Linear(C->classes).
"""
import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2          # 'same' (non-causal)
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation))
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):                                 # x: (B, C, T)
        h = self.act(self.conv1(x))
        h = self.drop(h)
        h = self.act(self.conv2(h))
        h = self.drop(h)
        return x + h


class TCNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, channels=128, num_blocks=4,
                 kernel_size=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, channels)
        self.ln_in = nn.LayerNorm(channels)
        self.blocks = nn.Sequential(*[
            TemporalBlock(channels, kernel_size, 2 ** i, dropout) for i in range(num_blocks)])
        self.ln_out = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x, lengths):
        h = self.ln_in(self.proj(x)).transpose(1, 2)      # (B, C, T)
        h = self.blocks(h)
        h = self.ln_out(h.transpose(1, 2))                 # (B, T, C)
        mask = (torch.arange(h.size(1), device=x.device).unsqueeze(0)
                < lengths.to(x.device).unsqueeze(1)).unsqueeze(-1).float()
        mean = (h * mask).sum(1) / lengths.to(x.device).unsqueeze(1).float()
        return self.head(self.drop(mean))

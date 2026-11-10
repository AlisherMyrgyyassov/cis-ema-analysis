"""Transformer-encoder classifier baseline (Vaswani et al., 2017).

Mirrors the canonical BiLSTM interface/protocol so it can be swapped in without
any other change: Linear(d->d_model) + LayerNorm -> sinusoidal positional
encoding -> N pre-LN Transformer encoder layers -> LayerNorm -> masked mean-pool
-> Linear(d_model -> num_classes). Padding is masked in attention and pooling.
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=512, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ln_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths):
        h = self.ln_in(self.proj(x))
        h = self.pos(h)
        T = x.size(1)
        key_pad = (torch.arange(T, device=x.device).unsqueeze(0)
                   >= lengths.to(x.device).unsqueeze(1))          # True = ignore
        h = self.enc(h, src_key_padding_mask=key_pad)
        mask = (~key_pad).unsqueeze(-1).float()
        mean = (h * mask).sum(1) / lengths.to(x.device).unsqueeze(1).float()
        return self.head(self.drop(mean))

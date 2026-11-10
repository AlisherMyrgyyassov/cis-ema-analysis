"""Canonical BiLSTM backbone (Experiment 0.4).

Architecture (fixed across Experiments 1-4):

    Linear(d -> 128) -> LayerNorm -> ReLU
    -> 2 x BiLSTM(128)   (explicitly stacked for per-layer probing)
    -> LayerNorm -> masked mean-pool
    -> Linear(256 -> 43)

For Experiment 3 probing, ``forward(..., return_hidden=True)`` additionally
returns intermediate representations at each stage:
    "proj"    (B, T, 128)   after the input projection
    "layer0"  (B, T, 256)   after the first BiLSTM
    "layer1"  (B, T, 256)   after the second BiLSTM
    "pooled"  (B, 256)      after masked mean-pool (before the head)
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SyllableBiLSTM(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, num_layers=2,
                 dropout=0.4):
        super().__init__()
        self.hidden = hidden
        self.num_layers = num_layers

        self.proj = nn.Linear(input_dim, hidden)
        self.ln_in = nn.LayerNorm(hidden)

        # Explicitly stacked BiLSTMs so intermediate layer outputs can be
        # extracted for Experiment 3 probing. Inter-layer dropout is applied
        # manually in ``encode`` (each individual LSTM keeps dropout=0).
        self.lstms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden if i == 0 else hidden * 2
            self.lstms.append(
                nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True))

        self.ln_out = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 2, num_classes)

    def encode(self, x, lengths, return_hidden=False):
        proj = torch.relu(self.ln_in(self.proj(x)))
        lengths_cpu = lengths.cpu()
        sorted_len, sort_idx = torch.sort(lengths_cpu, descending=True)
        _, unsort_idx = torch.sort(sort_idx)
        device = x.device

        h = proj[sort_idx.to(device)]
        layer_outs = []
        for i, lstm in enumerate(self.lstms):
            packed = pack_padded_sequence(h, sorted_len, batch_first=True,
                                          enforce_sorted=True)
            out, _ = lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)
            layer_outs.append(out[unsort_idx.to(device)])
            if i < self.num_layers - 1:
                h = self.drop(out)   # inter-layer dropout (matches V2)
            else:
                h = out

        final = self.ln_out(layer_outs[-1])
        if return_hidden:
            hidden = {"proj": proj, "layer0": layer_outs[0]}
            if len(layer_outs) > 1:
                hidden["layer1"] = layer_outs[-1]
            return final, hidden
        return final

    def forward(self, x, lengths, return_hidden=False):
        if return_hidden:
            final, hidden = self.encode(x, lengths, return_hidden=True)
        else:
            final = self.encode(x, lengths)

        mask = (torch.arange(final.size(1), device=final.device).unsqueeze(0)
                < lengths.to(final.device).unsqueeze(1)).float()
        mean = (final * mask.unsqueeze(-1)).sum(1) / lengths.to(final.device).unsqueeze(1).float()
        emb = self.drop(mean)
        logits = self.head(emb)

        if return_hidden:
            hidden["pooled"] = emb
            hidden["pooled_pre"] = mean
            return logits, hidden
        return logits


class SyllableBiLSTMAux(SyllableBiLSTM):
    """BiLSTM + an auxiliary regression head (used by Exp 5).

    Adds a linear head that regresses a small set of acoustic targets (e.g.
    per-utterance MFCC band means) from the pooled embedding, so an auxiliary
    MSE loss can force the EMA encoder to retain information it would otherwise
    ignore. ``forward`` returns (logits, aux) instead of logits alone.
    """
    def __init__(self, input_dim, num_classes, aux_dim, hidden=128,
                 num_layers=2, dropout=0.4):
        super().__init__(input_dim, num_classes, hidden=hidden,
                         num_layers=num_layers, dropout=dropout)
        self.head_aux = nn.Linear(hidden * 2, aux_dim)

    def forward(self, x, lengths):
        final = self.encode(x, lengths)
        mask = (torch.arange(final.size(1), device=final.device).unsqueeze(0)
                < lengths.to(final.device).unsqueeze(1)).float()
        mean = (final * mask.unsqueeze(-1)).sum(1) / lengths.to(final.device).unsqueeze(1).float()
        emb = self.drop(mean)
        logits = self.head(emb)
        aux = self.head_aux(emb)
        return logits, aux


class SyllableBiLSTMFrameAux(SyllableBiLSTM):
    """BiLSTM + a per-frame auxiliary regression head (Exp 5b).

    Unlike ``SyllableBiLSTMAux`` (which reads the pooled embedding that feeds
    the classifier), this head sits on the *pre-pooling* LSTM output at every
    time step, so the encoder is forced to track frame-level acoustic detail
    that masked mean-pooling would otherwise discard. ``forward`` returns
    ``(logits, frame_out)`` where ``frame_out`` is ``(B, T, aux_dim)`` aligned
    with the input time axis (padding must be masked by the caller).
    """
    def __init__(self, input_dim, num_classes, aux_dim, hidden=128,
                 num_layers=2, dropout=0.4):
        super().__init__(input_dim, num_classes, hidden=hidden,
                         num_layers=num_layers, dropout=dropout)
        self.head_frame = nn.Linear(hidden * 2, aux_dim)

    def forward(self, x, lengths):
        final = self.encode(x, lengths)
        frame_out = self.head_frame(final)
        mask = (torch.arange(final.size(1), device=final.device).unsqueeze(0)
                < lengths.to(final.device).unsqueeze(1)).float()
        mean = (final * mask.unsqueeze(-1)).sum(1) / lengths.to(final.device).unsqueeze(1).float()
        emb = self.drop(mean)
        logits = self.head(emb)
        return logits, frame_out

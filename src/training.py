"""Training / evaluation loop for the canonical BiLSTM (used by Exp 1-3).

Reuses the proven recipe from the earlier pipeline: Adam (lr 3e-3, wd 1e-4),
ReduceLROnPlateau on validation accuracy, bucketed batches with
``pack_padded_sequence`` (no padding leakage), early stopping with best-model
checkpoint on validation accuracy, optional label smoothing and balanced class
weights.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler


class UttDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [s.shape[0] for s in seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.from_numpy(self.seqs[i]), int(self.labels[i])


class BucketSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = np.argsort(lengths, kind="stable")

    def __iter__(self):
        batches = [self.order[i:i + self.batch_size]
                   for i in range(0, len(self.order), self.batch_size)]
        if self.shuffle:
            batches = [np.random.permutation(b) for b in batches]
            np.random.shuffle(batches)
        for b in batches:
            yield b.tolist()

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size


def collate(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    F = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, F)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = s
    return x, lengths, labels


def make_loader(seqs, labels, batch_size, shuffle):
    ds = UttDataset(seqs, labels)
    return DataLoader(ds, batch_sampler=BucketSampler(ds.lengths, batch_size, shuffle),
                      collate_fn=collate)


def evaluate_syl(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, lengths, labels in loader:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            logits = model(x, lengths)
            preds.append(logits.argmax(1).cpu())
            truths.append(labels.cpu())
    preds = torch.cat(preds).numpy()
    truths = torch.cat(truths).numpy()
    return preds, truths


def train_model(model, tr_loader, va_loader, device, lr=3e-3, weight_decay=1e-4,
                epochs=80, patience=15, label_smoothing=0.0, class_weight=None,
                verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=7)
    crit = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)

    best_val = -1.0
    best_state = None
    wait = patience

    for epoch in range(1, epochs + 1):
        model.train()
        for x, lengths, labels in tr_loader:
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            opt.zero_grad()
            loss = crit(model(x, lengths), labels)
            loss.backward()
            opt.step()

        va_preds, va_truths = evaluate_syl(model, va_loader, device)
        va_acc = float((va_preds == va_truths).mean())
        sched.step(va_acc)
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = patience
        else:
            wait -= 1
        if verbose and (epoch % 10 == 0 or wait <= 0 or epoch == epochs):
            print(f"  epoch {epoch:3d}  val_acc {va_acc:.4f}  best {best_val:.4f}")
        if wait <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val

"""Experiment 5 (MFCC-band auxiliary losses) - readout- and encoder-level.

Two auxiliary interventions on the EMA-only BiLSTM, both forcing the network to
retain spectral information (MFCC_2/5/6) that Experiment 4 showed is jointly
recoverable from EMA:

  mode="uttmean" (5a, readout level): a regression head reads the *pooled
      embedding that feeds the classifier* and predicts the per-utterance mean
      of AUX_COLS. ``frame_set="all"`` averages over every frame (matching
      Exp-4's all-frame MFCC treatment and the mean-pooled readout);
      ``frame_set="voiced"`` restricts the mean to voiced frames (robustness
      check for the "mixing silence/onset into the target" objection).
  mode="perframe" (5b, encoder level): a per-frame head on the *pre-pooling*
      LSTM output predicts the within-utterance z-scored trajectory of
      AUX_COLS at every time step (the same quantity Exp-4's joint ridge
      measured, now through a neural bottleneck instead of a linear one).

Loss = cross-entropy + lambda * MSE, same recipe / CV / seeds as Experiments
1-4 and the earlier Exp-5 confirmatory run.

For every configuration the auxiliary fit is now *measured* (train/val/test
aux R^2, plus per-run checkpoints), so a flat syllable-accuracy result can be
attributed to either (a) the aux signal never being learned or (b) the
information being genuinely redundant with the classifier's readout - instead
of being uninterpretable as before.

NOTE (implementation fix): aux targets are carried inside the dataset so they
stay aligned with their utterance through the bucketed sampler. The previous
version fetched targets by sequential chunking of ``aux_tr[i:i+batch]``, which
mismatches the shuffled batch order and effectively trained the MSE head
against *random* utterances.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

from src.config import RUNS_DIR, N_FOLDS, N_VAL_SUBJECTS, FOLD_SEED, SEEDS, DATA_DIR
from src.data.dataset import load_dataset, records_for, view, fit_stats, standardize
from src.data.cv import make_folds
from src.models.bilstm import SyllableBiLSTMAux, SyllableBiLSTMFrameAux
from src.training import BucketSampler
from src.eval import classification_metrics

AUX_COLS = ["MFCC_2", "MFCC_5", "MFCC_6"]
EMA_GROUP_IDS = ["G1", "G2", "G3", "G4"]
MIN_VOICED_FRAMES = 5


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def _frame_select(record, col_index, cols):
    idx = [col_index[c] for c in cols]
    return record["frame"][:, idx].astype(np.float64)


def _voiced_mask(record, col_index):
    return record["frame"][:, col_index["voiced"]] > 0.5


def record_uttmean(record, col_index, frame_set):
    """(mean (3,), valid) of AUX_COLS over all or voiced frames."""
    a = _frame_select(record, col_index, AUX_COLS)
    if frame_set == "all":
        m = np.ones(len(a), dtype=bool)
    elif frame_set == "voiced":
        m = _voiced_mask(record, col_index)
    else:
        raise ValueError(frame_set)
    if m.sum() < MIN_VOICED_FRAMES:
        return np.zeros(3), False
    return a[m].mean(0), True


def record_perframe(record, col_index):
    """(frames (T,3), chan_mask (T,3)) of within-utterance z-scored AUX_COLS.

    Each channel is z-scored across that utterance's own frames, matching the
    within-utterance standardization of Exp-4's ridge (speaker level/scale
    removed; only the time-course is the target). Frames where the channel has
    no variance are masked out of the loss.
    """
    a = _frame_select(record, col_index, AUX_COLS)
    mu = a.mean(0, keepdims=True)
    sd = a.std(0) + 1e-9
    z = (a - mu) / sd
    chan_mask = np.tile((a.std(0) > 1e-6)[None, :], (len(a), 1))
    return z.astype(np.float32), chan_mask


# --------------------------------------------------------------------------- #
# Datasets (targets aligned with utterances through the sampler)
# --------------------------------------------------------------------------- #
class UMeanAuxDataset(Dataset):
    """seq + scalar utterance target (+valid mask) + label, index-aligned."""

    def __init__(self, seqs, targets, valid, labels):
        self.seqs = seqs
        self.targets = targets
        self.valid = np.asarray(valid, dtype=bool)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [s.shape[0] for s in seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return (self.seqs[i], self.targets[i], bool(self.valid[i]),
                int(self.labels[i]))


class FrameAuxDataset(Dataset):
    """seq + per-frame target/mask + label, index-aligned."""

    def __init__(self, seqs, frames, masks, labels):
        self.seqs = seqs
        self.frames = frames
        self.masks = masks
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [s.shape[0] for s in seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return (self.seqs[i], self.frames[i], self.masks[i], int(self.labels[i]))


def _pad(batch_arrays, max_len):
    """Pad a list of (T, d) arrays to (B, max_len, d)."""
    B = len(batch_arrays)
    d = batch_arrays[0].shape[1]
    out = np.zeros((B, max_len, d), dtype=np.float32)
    for b, arr in enumerate(batch_arrays):
        out[b, :arr.shape[0]] = arr
    return out


def collate_umean(batch):
    seqs, tgt, valid, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    F = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, F)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = torch.from_numpy(s)
    return (x, lengths, torch.tensor(np.stack(tgt), dtype=torch.float32),
            torch.tensor(np.asarray(valid), dtype=torch.bool),
            torch.tensor(np.asarray(labels), dtype=torch.long))


def collate_frame(batch):
    seqs, frames, masks, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())
    F = seqs[0].shape[1]
    x = torch.zeros(len(seqs), max_len, F)
    fr = torch.zeros(len(seqs), max_len, len(AUX_COLS))
    mk = torch.zeros(len(seqs), max_len, len(AUX_COLS))
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = torch.from_numpy(s)
        fr[i, :s.shape[0]] = torch.from_numpy(frames[i])
        mk[i, :s.shape[0]] = torch.from_numpy(masks[i])
    return (x, lengths, fr, mk, torch.tensor(np.asarray(labels), dtype=torch.long))


def make_loader(ds, batch_size, shuffle, collate):
    return DataLoader(ds, batch_sampler=BucketSampler(ds.lengths, batch_size,
                                                      shuffle), collate_fn=collate)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _build_opt(model, device, lr=3e-3, weight_decay=1e-4, class_weight=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit_ce = nn.CrossEntropyLoss(weight=class_weight)
    crit_mse = nn.MSELoss()
    return opt, crit_ce, crit_mse


def train_uttmean(model, tr_loader, va_loader, device, lmbda, class_weight=None,
                  lr=3e-3, weight_decay=1e-4, epochs=80, patience=15,
                  verbose=True):
    opt, crit_ce, crit_mse = _build_opt(model, device, lr, weight_decay,
                                        class_weight)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=7)
    best_val = -1.0
    best_state = None
    wait = patience
    for epoch in range(1, epochs + 1):
        model.train()
        for x, lengths, tgt, valid, labels in tr_loader:
            x = x.to(device); lengths = lengths.to(device)
            tgt = tgt.to(device); labels = labels.to(device)
            opt.zero_grad()
            logits, aux = model(x, lengths)
            ce = crit_ce(logits, labels)
            if valid.any():
                mse = crit_mse(aux[valid], tgt[valid])
            else:
                mse = torch.zeros((), device=device)
            loss = ce + lmbda * mse
            loss.backward()
            opt.step()
        va_preds, va_truths = _predict(model, va_loader, device)
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


def train_perframe(model, tr_loader, va_loader, device, lmbda, class_weight=None,
                   lr=3e-3, weight_decay=1e-4, epochs=80, patience=15,
                   verbose=True):
    opt, crit_ce, crit_mse = _build_opt(model, device, lr, weight_decay,
                                        class_weight)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=7)
    best_val = -1.0
    best_state = None
    wait = patience
    for epoch in range(1, epochs + 1):
        model.train()
        for x, lengths, fr, mk, labels in tr_loader:
            x = x.to(device); lengths = lengths.to(device)
            fr = fr.to(device); mk = mk.to(device); labels = labels.to(device)
            opt.zero_grad()
            logits, frame_out = model(x, lengths)
            ce = crit_ce(logits, labels)
            m = mk.bool()
            if m.any():
                mse = ((frame_out[m] - fr[m]) ** 2).mean()
            else:
                mse = torch.zeros((), device=device)
            loss = ce + lmbda * mse
            loss.backward()
            opt.step()
        va_preds, va_truths = _predict(model, va_loader, device)
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


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _predict(model, loader, device):
    """(preds, truths) of the classification head."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch in loader:
            x, lengths, labels = batch[0], batch[1], batch[-1]
            x = x.to(device); lengths = lengths.to(device)
            if len(batch) == 5:                       # frame dataset -> (logits, _)
                logits, _ = model(x, lengths)
            else:                                     # umean dataset -> (logits, _)
                logits, _ = model(x, lengths)
            preds.append(logits.argmax(1).cpu())
            truths.append(labels.cpu())
    return torch.cat(preds).numpy(), torch.cat(truths).numpy()


def _eval_uttmean(model, loader, device):
    """Mean-over-channels aux R^2 over the loader's utterances."""
    model.eval()
    auxs, tgts, valids = [], [], []
    with torch.no_grad():
        for x, lengths, tgt, valid, _ in loader:
            x = x.to(device); lengths = lengths.to(device)
            _, aux = model(x, lengths)
            auxs.append(aux.cpu().numpy())
            tgts.append(tgt.numpy())
            valids.append(valid.numpy())
    aux = np.concatenate(auxs); tgt = np.concatenate(tgts)
    valid = np.concatenate(valids)
    r2 = []
    for c in range(len(AUX_COLS)):
        m = valid
        if m.sum() == 0:
            r2.append(0.0)
            continue
        p, t = aux[m, c], tgt[m, c]
        ss_res = float(((p - t) ** 2).sum())
        ss_tot = float(((t - t.mean()) ** 2).sum())
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
    return float(np.mean(r2))


def _eval_perframe(model, loader, device):
    """Per-utterance per-channel frame aux R^2 (median over utterances)."""
    model.eval()
    utt_r2 = []
    with torch.no_grad():
        for x, lengths, fr, mk, _ in loader:
            x = x.to(device); lengths = lengths.to(device)
            _, frame_out = model(x, lengths)
            fo = frame_out.cpu().numpy()
            fr = fr.numpy(); mk = mk.numpy()
            lengths = lengths.cpu().numpy()
            for b in range(len(lengths)):
                L = int(lengths[b])
                f = fr[b][:L]; p = fo[b][:L]; m = mk[b][:L].astype(bool)
                ch = []
                for c in range(len(AUX_COLS)):
                    mm = m[:, c]
                    if mm.sum() < 2:
                        continue
                    t = f[mm, c]; pr = p[mm, c]
                    ss_res = float(((pr - t) ** 2).sum())
                    ss_tot = float(((t - t.mean()) ** 2).sum())
                    ch.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
                if ch:
                    utt_r2.append(float(np.mean(ch)))
    return float(np.median(utt_r2)) if utt_r2 else 0.0


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _class_weights(y, device, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32).to(device)


def run(data_dir=None, out_dir=None, device="cuda", mode="uttmean",
        frame_set="all", lmbdas=(0.3, 1.0), seeds=SEEDS, fold=None,
        epochs=80, batch_size=32, verbose=True):
    data_dir = data_dir or DATA_DIR
    out_dir = out_dir or os.path.join(RUNS_DIR, f"exp5_{mode}_{frame_set}"
                                      if mode == "uttmean" else "exp5_perframe")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    records, col_index = load_dataset(data_dir)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)

    results = []
    for fold_id, fd in enumerate(folds):
        if fold is not None and fold_id != fold:
            continue
        tr = records_for(records, fd["train"])
        va = records_for(records, fd["val"])
        te = records_for(records, fd["test"])

        le = LabelEncoder().fit([r["syllable"] for r in tr])
        y_tr = le.transform([r["syllable"] for r in tr])
        y_va = le.transform([r["syllable"] for r in va])
        y_te = le.transform([r["syllable"] for r in te])
        num_classes = len(le.classes_)

        tr_seqs = view(tr, col_index, EMA_GROUP_IDS)
        va_seqs = view(va, col_index, EMA_GROUP_IDS)
        te_seqs = view(te, col_index, EMA_GROUP_IDS)
        mu, sd = fit_stats(tr_seqs)
        tr_std = standardize(tr_seqs, mu, sd)
        va_std = standardize(va_seqs, mu, sd)
        te_std = standardize(te_seqs, mu, sd)
        input_dim = tr_std[0].shape[1]

        cw = _class_weights(y_tr, device, num_classes)

        if mode == "uttmean":
            tr_tgt = [record_uttmean(r, col_index, frame_set) for r in tr]
            va_tgt = [record_uttmean(r, col_index, frame_set) for r in va]
            te_tgt = [record_uttmean(r, col_index, frame_set) for r in te]
            t_tr = np.stack([t[0] for t in tr_tgt])
            m_tr = t_tr.mean(0, keepdims=True)
            s_tr = t_tr.std(0, keepdims=True) + 1e-6
            targets = lambda tgt_list: np.stack([t[0] for t in tgt_list])
            scales = lambda arr: (arr - m_tr) / s_tr
            tr_ds = UMeanAuxDataset(tr_std, scales(targets(tr_tgt)),
                                    [t[1] for t in tr_tgt], y_tr)
            va_ds = UMeanAuxDataset(va_std, scales(targets(va_tgt)),
                                    [t[1] for t in va_tgt], y_va)
            te_ds = UMeanAuxDataset(te_std, scales(targets(te_tgt)),
                                    [t[1] for t in te_tgt], y_te)
            train_fn = train_uttmean
            eval_aux = _eval_uttmean
            ckpt_extra = {"mode": mode, "frame_set": frame_set,
                          "target_mu": m_tr, "target_sd": s_tr}
        elif mode == "perframe":
            tr_ds = FrameAuxDataset(tr_std, *zip(*[record_perframe(r, col_index)
                                                   for r in tr]), y_tr)
            va_ds = FrameAuxDataset(va_std, *zip(*[record_perframe(r, col_index)
                                                   for r in va]), y_va)
            te_ds = FrameAuxDataset(te_std, *zip(*[record_perframe(r, col_index)
                                                   for r in te]), y_te)
            train_fn = train_perframe
            eval_aux = _eval_perframe
            ckpt_extra = {"mode": mode}
        else:
            raise ValueError(mode)

        tr_loader = make_loader(tr_ds, batch_size, True,
                                collate_umean if mode == "uttmean" else collate_frame)
        va_loader = make_loader(va_ds, batch_size, False,
                                collate_umean if mode == "uttmean" else collate_frame)
        te_loader = make_loader(te_ds, batch_size, False,
                                collate_umean if mode == "uttmean" else collate_frame)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            for lmbda in lmbdas:
                if mode == "uttmean":
                    model = SyllableBiLSTMAux(input_dim, num_classes,
                                              len(AUX_COLS)).to(device)
                else:
                    model = SyllableBiLSTMFrameAux(input_dim, num_classes,
                                                   len(AUX_COLS)).to(device)
                model, best_val = train_fn(model, tr_loader, va_loader, device,
                                           lmbda, class_weight=cw,
                                           epochs=epochs, verbose=verbose)
                te_preds, te_truths = _predict(model, te_loader, device)
                m = classification_metrics(te_truths, te_preds,
                                           classes=le.classes_.tolist())
                diag = {"aux_r2_train": eval_aux(model, tr_loader, device),
                        "aux_r2_val": eval_aux(model, va_loader, device),
                        "aux_r2_test": eval_aux(model, te_loader, device)}
                results.append({"fold": fold_id, "seed": seed, "lmbda": lmbda,
                                "mode": mode, "frame_set": frame_set,
                                "val_acc": float(best_val),
                                "test_acc": m["accuracy"], "test_f1": m["macro_f1"],
                                **diag})
                ckpt_dir = os.path.join(out_dir, "models")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model": {k: v.cpu().clone()
                                      for k, v in model.state_dict().items()},
                            "classes": le.classes_.tolist(),
                            "input_dim": input_dim, "num_classes": num_classes,
                            "fold": fold_id, "seed": seed, "lmbda": lmbda,
                            **ckpt_extra},
                           os.path.join(ckpt_dir,
                                        f"fold{fold_id}_seed{seed}_lmbda{lmbda}.pth"))
                if verbose:
                    print(f"[fold {fold_id} seed {seed} lmbda {lmbda}] "
                          f"val {best_val:.4f} test_acc {m['accuracy']:.4f} "
                          f"test_f1 {m['macro_f1']:.4f} auxR2 {diag}")

    summary = {}
    for lmbda in lmbdas:
        rs = [r for r in results if r["lmbda"] == lmbda]
        if not rs:
            continue
        accs = [r["test_acc"] for r in rs]
        f1s = [r["test_f1"] for r in rs]
        summary[str(lmbda)] = {
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)),
            "test_f1_std": float(np.std(f1s)),
            "aux_r2_train_mean": float(np.mean([r["aux_r2_train"] for r in rs])),
            "aux_r2_val_mean": float(np.mean([r["aux_r2_val"] for r in rs])),
            "aux_r2_test_mean": float(np.mean([r["aux_r2_test"] for r in rs])),
            "n": len(rs),
        }
    out = {"aux_cols": AUX_COLS, "mode": mode, "frame_set": frame_set,
           "summary": summary, "runs": results}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n==== Experiment 5 ({mode}"
          + (f", frame_set={frame_set}" if mode == "uttmean" else "") + ") ====")
    for lmbda in lmbdas:
        s = summary.get(str(lmbda))
        if not s:
            continue
        print(f"  lambda {lmbda}: test_acc {s['test_acc_mean']*100:.2f} +- "
              f"{s['test_acc_std']*100:.2f}  test_f1 {s['test_f1_mean']*100:.2f} "
              f"auxR2(val) {s['aux_r2_val_mean']:.3f} auxR2(test) {s['aux_r2_test_mean']:.3f}")
    return summary

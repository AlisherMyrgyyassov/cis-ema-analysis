"""Experiment 6: distillation into the EMA-only student.

Two capabilities:

1. ``eval_dir`` -- evaluate a directory of saved checkpoints (per-run + per-fold
   logit-ensemble). This is the no-training diagnostic / strong baseline.

2. ``train_kd_cv`` -- train an EMA-only ``SyllableBiLSTM`` under the locked
   subject-disjoint CV with knowledge distillation from a frozen, same-fold
   teacher (the Exp-1 full 106-D model by default).

Teacher targets are computed once per (fold, seed) and carried *inside* the
dataset so they stay aligned with each utterance through the bucketed sampler
(the same alignment fix used in Exp 5). The student is EMA-only at inference;
the teacher only shapes training.

Loss = (1-alpha) * CE + alpha * T^2 * KL(student/T || teacher/T)
       + lambda_feat * MSE(student_pooled, teacher_pooled)
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import EMA_GROUPS, GROUP_IDS
from src.data.dataset import records_for, view, fit_stats, standardize
from src.models.bilstm import SyllableBiLSTM
from src.eval import classification_metrics
from src.training import BucketSampler, make_loader, evaluate_syl


# --------------------------------------------------------------------------- #
# Checkpoints / inference
# --------------------------------------------------------------------------- #
def load_ckpt(ckpt_path, device, num_layers=2, hidden=128, dropout=0.4):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SyllableBiLSTM(ckpt["input_dim"], ckpt["num_classes"],
                           num_layers=num_layers, hidden=hidden, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def logits_for(model, seqs, device, batch_size=256, return_pooled=False):
    model.eval()
    logits_all, pooled_all = [], []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            lengths = torch.tensor([s.shape[0] for s in chunk], dtype=torch.long)
            mx = int(lengths.max())
            Fd = chunk[0].shape[1]
            x = torch.zeros(len(chunk), mx, Fd)
            for b, s in enumerate(chunk):
                x[b, :s.shape[0]] = torch.from_numpy(s)
            x, lengths = x.to(device), lengths.to(device)
            if return_pooled:
                logits, hidden = model(x, lengths, return_hidden=True)
                pooled_all.append(hidden["pooled_pre"].cpu())
            else:
                logits = model(x, lengths)
            logits_all.append(logits.cpu())
    logits = torch.cat(logits_all).numpy()
    if return_pooled:
        return logits, torch.cat(pooled_all).numpy()
    return logits


def _labels_int(records, classes):
    cls2i = {c: i for i, c in enumerate(classes)}
    return np.array([cls2i[r["syllable"]] for r in records], dtype=np.int64)


def prepare_fold(records, col_index, fold, groups):
    tr = records_for(records, fold["train"])
    va = records_for(records, fold["val"])
    te = records_for(records, fold["test"])
    tr_seqs = view(tr, col_index, groups)
    va_seqs = view(va, col_index, groups)
    te_seqs = view(te, col_index, groups)
    mu, sd = fit_stats(tr_seqs)
    tr_std = standardize(tr_seqs, mu, sd)
    va_std = standardize(va_seqs, mu, sd)
    te_std = standardize(te_seqs, mu, sd)
    classes = sorted({r["syllable"] for r in tr})
    return dict(tr=tr, va=va, te=te, tr_std=tr_std, va_std=va_std, te_std=te_std,
                y_tr=_labels_int(tr, classes), y_va=_labels_int(va, classes),
                y_te=_labels_int(te, classes), classes=classes)


# --------------------------------------------------------------------------- #
# Diagnostic: evaluate a checkpoint directory (+ per-fold ensemble)
# --------------------------------------------------------------------------- #
def eval_dir(records, col_index, folds, ckpt_dir, groups, device,
             num_layers=2, hidden=128, dropout=0.4, tag="_model",
             seeds=(0, 1), save=None):
    per_run, ens_by_fold = [], {}
    for fold_id, fold in enumerate(folds):
        pf = prepare_fold(records, col_index, fold, groups)
        ens_logits = []
        for seed in seeds:
            path = os.path.join(ckpt_dir, f"fold{fold_id}_seed{seed}{tag}.pth")
            if not os.path.isfile(path):
                continue
            model, ckpt = load_ckpt(path, device, num_layers, hidden, dropout)
            logits = logits_for(model, pf["te_std"], device)
            m = classification_metrics(pf["y_te"], logits.argmax(1), classes=pf["classes"])
            per_run.append({"fold": fold_id, "seed": seed,
                            "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
            ens_logits.append(logits)
        if ens_logits:
            ens = np.mean(ens_logits, axis=0)
            m = classification_metrics(pf["y_te"], ens.argmax(1), classes=pf["classes"])
            ens_by_fold[fold_id] = {"test_acc": m["accuracy"], "test_f1": m["macro_f1"]}

    def _agg(rows):
        if not rows:
            return {}
        accs = [r["test_acc"] for r in rows]
        f1s = [r["test_f1"] for r in rows]
        return {"acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "n": len(rows)}

    out = {"per_run": per_run, "summary_runs": _agg(per_run),
           "ens_by_fold": ens_by_fold, "summary_ensemble": _agg(list(ens_by_fold.values()))}
    if save:
        with open(save, "w") as f:
            json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# KD dataset / loader
# --------------------------------------------------------------------------- #
class KDDataset(Dataset):
    """Sequence + teacher logits + teacher pooled features + label, index-aligned."""

    def __init__(self, seqs, t_logits, t_pooled, labels):
        self.seqs = seqs
        self.t_logits = np.asarray(t_logits, dtype=np.float32)
        self.t_pooled = None if t_pooled is None else np.asarray(t_pooled, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [s.shape[0] for s in seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        item = [self.seqs[i], self.t_logits[i], self.labels[i]]
        if self.t_pooled is not None:
            item.append(self.t_pooled[i])
        return tuple(item)


def collate_kd(batch):
    seqs = [b[0] for b in batch]
    tlog = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    mx = int(lengths.max())
    Fd = seqs[0].shape[1]
    x = torch.zeros(len(seqs), mx, Fd)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = torch.from_numpy(s)
    out = [x, lengths, tlog, labels]
    if len(batch[0]) > 3:
        out.append(torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32))
    return tuple(out)


def make_kd_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_sampler=BucketSampler(ds.lengths, batch_size, shuffle),
                      collate_fn=collate_kd)


# --------------------------------------------------------------------------- #
# KD training
# --------------------------------------------------------------------------- #
def train_kd_cv(records, col_index, folds, teacher_dir, out_dir, name,
                device="cuda", seed_list=(0, 1), temperature=4.0, alpha=0.9,
                lambda_feat=0.0, teacher_tag="_model", teacher_groups=None,
                teacher_seeds=None, ema_groups=EMA_GROUPS, epochs=80, patience=15,
                batch_size=32, lr=3e-3, weight_decay=1e-4, dropout=0.4,
                num_layers=2, hidden=128, class_weight=True, verbose=True,
                fold=None, save_ckpts=False, shuffle_teacher=False):
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    if teacher_groups is None:
        teacher_groups = GROUP_IDS
    results = []

    for fold_id, fd in enumerate(folds):
        if fold is not None and fold_id != fold:
            continue
        pf = prepare_fold(records, col_index, fd, ema_groups)
        num_classes = len(pf["classes"])
        input_dim = pf["tr_std"][0].shape[1]

        cw = None
        if class_weight:
            counts = np.bincount(pf["y_tr"], minlength=num_classes).astype(np.float32)
            cw = torch.tensor(counts.sum() / (num_classes * counts),
                              dtype=torch.float32).to(device)

        va_loader = make_loader(pf["va_std"], pf["y_va"], batch_size, False)
        te_loader = make_loader(pf["te_std"], pf["y_te"], batch_size, False)

        for seed in seed_list:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # ---- frozen teacher(s): average logits/pooled over teacher_seeds ----
            tseeds = teacher_seeds if teacher_seeds else (None,)
            t_logits_list, t_pool_list = [], []
            for tseed in tseeds:
                s = seed if tseed is None else tseed
                tpath = os.path.join(teacher_dir, f"fold{fold_id}_seed{s}{teacher_tag}.pth")
                tmodel, tckpt = load_ckpt(tpath, device, num_layers, hidden, dropout)
                t_classes = tckpt["classes"]
                t_groups = tckpt.get("groups", teacher_groups)
                tprep = prepare_fold(records, col_index, fd, t_groups)
                tl, tp = logits_for(tmodel, tprep["tr_std"], device, return_pooled=True)
                tidx = {c: i for i, c in enumerate(t_classes)}
                order = [tidx[c] for c in pf["classes"]]
                t_logits_list.append(tl[:, order])
                t_pool_list.append(tp)
                del tmodel
            t_logits_tr = (np.mean(t_logits_list, axis=0) if len(t_logits_list) > 1
                           else t_logits_list[0])
            t_pool_tr = np.mean(t_pool_list, axis=0)
            if shuffle_teacher:
                # control: keep the teacher distribution but break its pairing
                rng = np.random.RandomState(1234 + fold_id)
                perm = rng.permutation(len(t_logits_tr))
                t_logits_tr = t_logits_tr[perm]
                t_pool_tr = t_pool_tr[perm]

            ds = KDDataset(pf["tr_std"], t_logits_tr,
                           t_pool_tr if lambda_feat > 0 else None, pf["y_tr"])
            tr_loader = make_kd_loader(ds, batch_size, True)

            model = SyllableBiLSTM(input_dim, num_classes, num_layers=num_layers,
                                   hidden=hidden, dropout=dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                               factor=0.5, patience=7)
            crit = nn.CrossEntropyLoss(weight=cw)
            mse = nn.MSELoss()

            best_val, best_state, wait = -1.0, None, patience
            for epoch in range(1, epochs + 1):
                model.train()
                for batch in tr_loader:
                    if lambda_feat > 0:
                        x, lengths, tlog, labels, tpool = batch
                        tpool = tpool.to(device)
                    else:
                        x, lengths, tlog, labels = batch
                    x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
                    tlog = tlog.to(device)
                    opt.zero_grad()
                    logits, hid = model(x, lengths, return_hidden=True)
                    ce = crit(logits, labels)
                    p_t = torch.softmax(tlog / temperature, dim=1)
                    p_s = torch.log_softmax(logits / temperature, dim=1)
                    kld = F.kl_div(p_s, p_t, reduction="batchmean")
                    loss = (1 - alpha) * ce + alpha * (temperature ** 2) * kld
                    if lambda_feat > 0:
                        loss = loss + lambda_feat * mse(hid["pooled_pre"], tpool)
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
                    print(f"  [f{fold_id} s{seed}] ep {epoch:3d} val {va_acc:.4f} best {best_val:.4f}")
                if wait <= 0:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            te_preds, te_truths = evaluate_syl(model, te_loader, device)
            m = classification_metrics(te_truths, te_preds, classes=pf["classes"])
            results.append({"fold": fold_id, "seed": seed, "val_acc": float(best_val),
                            "test_acc": m["accuracy"], "test_f1": m["macro_f1"],
                            "temperature": temperature, "alpha": alpha,
                            "lambda_feat": lambda_feat})
            if save_ckpts:
                torch.save({"model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                            "input_dim": input_dim, "num_classes": num_classes,
                            "classes": pf["classes"], "groups": ema_groups,
                            "fold": fold_id, "seed": seed},
                           os.path.join(out_dir, f"fold{fold_id}_seed{seed}_model.pth"))
            if verbose:
                print(f"[KD f{fold_id} s{seed} T{temperature} a{alpha} lf{lambda_feat}] "
                      f"val {best_val:.4f} test_acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f}")

    summary = summarize(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"name": name, "temperature": temperature, "alpha": alpha,
                   "lambda_feat": lambda_feat, "teacher_dir": teacher_dir,
                   "summary": summary, "runs": results}, f, indent=2)
    return summary, results


def summarize(results):
    if not results:
        return {}
    accs = [r["test_acc"] for r in results]
    f1s = [r["test_f1"] for r in results]
    return {"test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
            "n": len(results)}

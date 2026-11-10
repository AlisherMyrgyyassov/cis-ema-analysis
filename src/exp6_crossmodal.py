"""Experiment 6B: cross-modal distillation from the fine-tuned HuBERT audio
teacher into the EMA-only student.

This is the "distillation-based articulatory encoder" recipe. The audio teacher
is the same-fold fine-tuned HuBERT-large checkpoint from Exp 1.5. We use, as
training-only targets:

  - logit KD:    task-level soft labels from HuBERT's 43-way head.
  - feature KD:  HuBERT's pooled representation (after its 256-d projector),
                 matched by the student's pooled EMA embedding.
  - contrastive: InfoNCE alignment between the student EMA embedding and the
                 HuBERT embedding (in-batch negatives).

The student is EMA-only at inference; the microphone/HuBERT is used only to
shape training. Teacher outputs are precomputed once per fold and cached.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.config import EMA_GROUPS
from src.data.dataset import records_for, view, fit_stats, standardize
from src.models.bilstm import SyllableBiLSTM
from src.eval import classification_metrics
from src.training import BucketSampler

TARGET_SR = 16000


# --------------------------------------------------------------------------- #
# Teacher: HuBERT audio outputs
# --------------------------------------------------------------------------- #
def load_hubert(ckpt_path, device):
    from transformers import (AutoConfig, HubertForSequenceClassification,
                              Wav2Vec2ForSequenceClassification)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mid = ck["model_id"]
    cfg = AutoConfig.from_pretrained(mid)
    cfg.num_labels = len(ck["classes"])
    cls = (HubertForSequenceClassification if cfg.model_type == "hubert"
           else Wav2Vec2ForSequenceClassification)
    model = cls(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck["classes"], mid


def _masked_mean(hidden, attn):
    if attn is None:
        return hidden.mean(1)
    m = attn.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


def hubert_outputs(model, wavs, processor, device, batch_size=8):
    """(logits [N,43], pooled [N,256]) for a list of 16 kHz waveforms."""
    logits_all, pooled_all = [], []
    with torch.no_grad():
        for i in range(0, len(wavs), batch_size):
            chunk = [np.asarray(w, dtype=np.float32) for w in wavs[i:i + batch_size]]
            enc = processor(chunk, sampling_rate=TARGET_SR, return_tensors="pt",
                            padding=True, return_attention_mask=True)
            iv = enc["input_values"].to(device)
            attn = enc.get("attention_mask")
            attn = attn.to(device) if attn is not None else None
            out = model(input_values=iv, attention_mask=attn)
            logits_all.append(out.logits.cpu())
            hs = model.hubert(input_values=iv, attention_mask=attn).last_hidden_state
            # frame-level mask from sample-level lengths
            if attn is not None:
                in_len = attn.sum(-1).long()
                feat_len = model.hubert._get_feat_extract_output_lengths(in_len)
                Tf = hs.size(1)
                fmask = (torch.arange(Tf, device=device).unsqueeze(0)
                         < feat_len.unsqueeze(1))
            else:
                fmask = None
            pooled = model.projector(_masked_mean(hs, fmask))
            pooled_all.append(pooled.cpu())
    return torch.cat(logits_all).numpy(), torch.cat(pooled_all).numpy()


def precompute_hubert(records, folds, hubert_dir, cache_dir, device,
                      seed=0, processor=None):
    """Cache HuBERT logits+pooled for every record, per fold, aligned to records."""
    from src.models.ssl import default_processor
    from src.data.acoustic import cache_audio, TARGET_SR as _sr
    from src.config import DATA_DIR, RAW_DIR
    os.makedirs(cache_dir, exist_ok=True)
    if processor is None:
        processor = default_processor()
    audio = cache_audio(records, DATA_DIR, RAW_DIR)
    # keep only records that have audio, but preserve record order for those
    sources = [str(r["source"]) for r in records]
    wavs = [audio.get(s) for s in sources]
    out = {}
    for fold_id in range(len(folds)):
        path = os.path.join(cache_dir, f"hubert_fold{fold_id}_seed{seed}.npz")
        if os.path.isfile(path):
            z = np.load(path)
            out[fold_id] = {"logits": z["logits"], "pooled": z["pooled"],
                            "classes": list(z["classes"])}
            print(f"[hubert cache] fold {fold_id}: loaded {path}")
            continue
        ckpt = os.path.join(hubert_dir, f"fold{fold_id}_seed{seed}_model.pth")
        model, classes, mid = load_hubert(ckpt, device)
        # valid utterances = those with audio
        idx = [i for i, w in enumerate(wavs) if w is not None]
        sub = [wavs[i] for i in idx]
        print(f"[hubert] fold {fold_id}: embedding {len(sub)} utterances ...")
        lg, pl = hubert_outputs(model, sub, processor, device)
        L = np.full((len(records), len(classes)), np.nan, dtype=np.float32)
        P = np.full((len(records), pl.shape[1]), np.nan, dtype=np.float32)
        L[idx] = lg
        P[idx] = pl
        np.savez(path, logits=L, pooled=P, classes=np.array(classes))
        out[fold_id] = {"logits": L, "pooled": P, "classes": classes}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[hubert cache] fold {fold_id}: saved {path}")
    return out


# --------------------------------------------------------------------------- #
# Student dataset (carries index-aligned teacher targets)
# --------------------------------------------------------------------------- #
class XModalDataset(Dataset):
    def __init__(self, seqs, t_logits, t_pooled, labels, use_feat):
        self.seqs = seqs
        self.t_logits = t_logits
        self.t_pooled = t_pooled if use_feat else None
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [s.shape[0] for s in seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        if self.t_pooled is None:
            return (torch.from_numpy(self.seqs[i]), self.t_logits[i], self.labels[i])
        return (torch.from_numpy(self.seqs[i]), self.t_logits[i],
                self.t_pooled[i], self.labels[i])


def collate_x(batch):
    seqs = [b[0] for b in batch]
    tlog = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    labels = torch.tensor([b[-1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    mx = int(lengths.max())
    Fd = seqs[0].shape[1]
    x = torch.zeros(len(seqs), mx, Fd)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = s
    out = [x, lengths, tlog, labels]
    if len(batch[0]) == 4:
        # return [x, lengths, tpooled, tlogits, labels] to match training unpack
        out.insert(2, torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32))
    return tuple(out)


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_sampler=BucketSampler(ds.lengths, batch_size, shuffle),
                      collate_fn=collate_x)


class EMAKD(SyllableBiLSTM):
    """EMA student with an optional projection head for cross-modal alignment."""

    def __init__(self, input_dim, num_classes, proj_dim=256, hidden=128,
                 num_layers=2, dropout=0.4):
        super().__init__(input_dim, num_classes, hidden=hidden,
                         num_layers=num_layers, dropout=dropout)
        self.proj_modal = nn.Linear(hidden * 2, proj_dim)

    def forward(self, x, lengths, return_hidden=False):
        logits, hidden = super().forward(x, lengths, return_hidden=True)
        z = F.normalize(self.proj_modal(hidden["pooled_pre"]), dim=-1)
        if return_hidden:
            return logits, hidden, z
        return logits, z


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
    classes = sorted({r["syllable"] for r in tr})
    return dict(tr=tr, va=va, te=te,
                tr_std=standardize(tr_seqs, mu, sd),
                va_std=standardize(va_seqs, mu, sd),
                te_std=standardize(te_seqs, mu, sd),
                y_tr=_labels_int(tr, classes), y_va=_labels_int(va, classes),
                y_te=_labels_int(te, classes), classes=classes)


def _index_by_source(all_records):
    return {str(r["source"]): i for i, r in enumerate(all_records)}


def _targets_for(src_pos, split_records, hubert):
    """Collect teacher logits/pooled for split_records from the all-records arrays."""
    idx = [src_pos[str(r["source"])] for r in split_records]
    return hubert["logits"][idx], hubert["pooled"][idx]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_crossmodal(records, col_index, folds, hubert_cache, out_dir, name,
                     device="cuda", seed_list=(0, 1), temperature=4.0, alpha=0.5,
                     lambda_feat=0.0, lambda_con=0.0, epochs=80, patience=15,
                     batch_size=32, lr=3e-3, weight_decay=1e-4, dropout=0.4,
                     num_layers=2, hidden=128, class_weight=True, verbose=True):
    device = torch.device(device)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    src_pos = _index_by_source(records)

    for fold_id, fd in enumerate(folds):
        pf = prepare_fold(records, col_index, fd, EMA_GROUPS)
        num_classes = len(pf["classes"])
        input_dim = pf["tr_std"][0].shape[1]
        hub = hubert_cache[fold_id]
        hub_classes = hub["classes"]
        # reindex hubert logits to student order
        hidx = {c: i for i, c in enumerate(hub_classes)}
        order = [hidx[c] for c in pf["classes"]]

        def targets(split_recs):
            lg, pl = _targets_for(src_pos, split_recs, hub)
            return lg, pl

        lg_tr, pl_tr = targets(pf["tr"])
        lg_va, pl_va = targets(pf["va"])
        lg_tr = lg_tr[:, order]
        lg_va = lg_va[:, order]
        # rows without audio -> drop (NaN)
        good_tr = ~np.isnan(lg_tr).any(1)
        good_va = ~np.isnan(lg_va).any(1)
        tr_seqs = [pf["tr_std"][i] for i in range(len(pf["tr"])) if good_tr[i]]
        tr_y = pf["y_tr"][good_tr]
        tr_lg = lg_tr[good_tr]
        tr_pl = pl_tr[good_tr]

        cw = None
        if class_weight:
            counts = np.bincount(tr_y, minlength=num_classes).astype(np.float32)
            cw = torch.tensor(counts.sum() / (num_classes * counts),
                              dtype=torch.float32).to(device)
        va_loader = make_va_loader(pf["va_std"], pf["y_va"], batch_size)
        te_loader = make_va_loader(pf["te_std"], pf["y_te"], batch_size)

        for seed in seed_list:
            torch.manual_seed(seed)
            np.random.seed(seed)
            use_feat = lambda_feat > 0 or lambda_con > 0
            ds = XModalDataset(tr_seqs, tr_lg, tr_pl, tr_y, use_feat)
            tr_loader = make_loader(ds, batch_size, True)

            model = EMAKD(input_dim, num_classes, hidden=hidden,
                          num_layers=num_layers, dropout=dropout).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                               factor=0.5, patience=7)
            crit = nn.CrossEntropyLoss(weight=cw)
            mse = nn.MSELoss()

            best_val, best_state, wait = -1.0, None, patience
            for epoch in range(1, epochs + 1):
                model.train()
                for batch in tr_loader:
                    if use_feat:
                        x, lengths, tpool, tlog, labels = batch
                        tpool = tpool.to(device)
                    else:
                        x, lengths, tlog, labels = batch
                    x, lengths = x.to(device), lengths.to(device)
                    labels, tlog = labels.to(device), tlog.to(device)
                    opt.zero_grad()
                    logits, hid, z = model(x, lengths, return_hidden=True)
                    ce = crit(logits, labels)
                    p_t = torch.softmax(tlog / temperature, dim=1)
                    p_s = torch.log_softmax(logits / temperature, dim=1)
                    kld = F.kl_div(p_s, p_t, reduction="batchmean")
                    loss = (1 - alpha) * ce + alpha * (temperature ** 2) * kld
                    if use_feat:
                        # teacher pooled is 256-d; student pre-pool 256-d
                        loss = loss + lambda_feat * mse(hid["pooled_pre"], tpool)
                    if lambda_con > 0:
                        tz = F.normalize(tpool, dim=-1)
                        logits_c = z @ tz.t() / 0.1
                        tgt = torch.arange(len(z), device=device)
                        loss = loss + lambda_con * F.cross_entropy(logits_c, tgt)
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
                    print(f"  [f{fold_id} s{seed}] ep {epoch:3d} val {va_acc:.4f} best {best_val:.4f}")
                if wait <= 0:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)
            te_preds, te_truths = _predict(model, te_loader, device)
            m = classification_metrics(te_truths, te_preds, classes=pf["classes"])
            results.append({"fold": fold_id, "seed": seed, "val_acc": float(best_val),
                            "test_acc": m["accuracy"], "test_f1": m["macro_f1"]})
            if verbose:
                print(f"[XKD f{fold_id} s{seed}] val {best_val:.4f} "
                      f"test_acc {m['accuracy']:.4f} f1 {m['macro_f1']:.4f}")

    summary = _summ(results)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"name": name, "temperature": temperature, "alpha": alpha,
                   "lambda_feat": lambda_feat, "lambda_con": lambda_con,
                   "summary": summary, "runs": results}, f, indent=2)
    return summary, results


def make_va_loader(seqs, labels, batch_size):
    ds = _PlainDataset(seqs, labels)
    return DataLoader(ds, batch_sampler=BucketSampler(ds.lengths, batch_size, False),
                      collate_fn=_collate_plain)


class _PlainDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [s.shape[0] for s in seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return torch.from_numpy(self.seqs[i]), int(self.labels[i])


def _collate_plain(batch):
    seqs = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    mx = int(lengths.max())
    Fd = seqs[0].shape[1]
    x = torch.zeros(len(seqs), mx, Fd)
    for i, s in enumerate(seqs):
        x[i, :s.shape[0]] = s
    return x, lengths, labels


def _predict(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for x, lengths, labels in loader:
            x, lengths = x.to(device), lengths.to(device)
            out = model(x, lengths)
            logits = out[0] if isinstance(out, tuple) else out
            preds.append(logits.argmax(1).cpu())
            truths.append(labels)
    return torch.cat(preds).numpy(), torch.cat(truths).numpy()


def _summ(results):
    if not results:
        return {}
    accs = [r["test_acc"] for r in results]
    f1s = [r["test_f1"] for r in results]
    return {"test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
            "n": len(results)}

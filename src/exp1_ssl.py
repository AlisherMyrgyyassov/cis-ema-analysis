"""Experiment 1.5: SSL audio ceiling (wav2vec2 / HuBERT / XLSR).

Fine-tunes a pretrained SSL sequence classifier on isolated-syllable raw audio
under the *same* sex-stratified, subject-disjoint 6-fold CV as the BiLSTM
ceilings. The CNN feature encoder stays frozen; SpecAugment is disabled.

Example:
  python -m scripts.run_exp1_ssl --model auto --seeds 0 --fold -1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from src.config import DATA_DIR, RAW_DIR, RUNS_DIR, N_FOLDS, N_VAL_SUBJECTS, FOLD_SEED
from src.data.dataset import load_dataset, records_for
from src.data.cv import make_folds
from src.data.acoustic import cache_audio, TARGET_SR
from src.models.ssl import load_ssl_model, default_processor
from src.eval import classification_metrics, summarize

TARGET_SR = 16000


class AudioUttDataset(Dataset):
    def __init__(self, wavs, labels):
        self.wavs = wavs
        self.labels = np.asarray(labels, dtype=np.int64)
        self.lengths = [int(w.shape[0]) for w in wavs]

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, i):
        wav = np.asarray(self.wavs[i], dtype=np.float32)
        min_samples = int(0.25 * TARGET_SR)
        if wav.shape[0] < min_samples:
            wav = np.pad(wav, (0, min_samples - wav.shape[0]))
        return wav, int(self.labels[i])


class BucketSampler:
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


def make_collate(processor):
    def collate(batch):
        wavs = [np.asarray(b[0], dtype=np.float32) for b in batch]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = processor(wavs, sampling_rate=TARGET_SR, return_tensors="pt",
                        padding=True, return_attention_mask=True)
        return enc["input_values"], enc.get("attention_mask"), labels
    return collate


def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for input_values, attn, labels in loader:
            input_values = input_values.to(device)
            labels = labels.to(device)
            kw = {}
            if attn is not None:
                kw["attention_mask"] = attn.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_values=input_values, **kw).logits
            preds.append(logits.argmax(1).cpu())
            truths.append(labels.cpu())
    return torch.cat(preds).numpy(), torch.cat(truths).numpy()


def train_one_fold(model, tr_loader, va_loader, device, lr=3e-5,
                   weight_decay=0.01, epochs=20, patience=5, grad_accum=2):
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    wait = patience

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running, n_steps = 0.0, 0
        for step, (input_values, attn, labels) in enumerate(tr_loader, start=1):
            input_values = input_values.to(device)
            labels = labels.to(device)
            kw = {}
            if attn is not None:
                kw["attention_mask"] = attn.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_values=input_values, **kw).logits
                loss = crit(logits, labels) / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            n_steps += 1
            if step % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

        va_preds, va_truths = evaluate(model, va_loader, device, use_amp)
        va_acc = float((va_preds == va_truths).mean())
        print(f"  epoch {epoch:02d}  loss {running / max(n_steps, 1):.4f}  val_acc {va_acc:.4f}")
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = patience
        else:
            wait -= 1
            if wait <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def run(data_dir=DATA_DIR, raw_dir=RAW_DIR, out_dir=None, model_arg="auto",
        seeds=(0,), fold=-1, device="cuda", epochs=20, batch_size=8,
        grad_accum=2, lr=3e-5, weight_decay=0.01, patience=5):
    device = torch.device(device)
    use_amp = device.type == "cuda"
    out_dir = out_dir or os.path.join(RUNS_DIR, "exp1_ssl")
    os.makedirs(out_dir, exist_ok=True)

    records, _ = load_dataset(data_dir)
    subjects = sorted({r["subj"] for r in records})
    sex_map = {r["subj"]: r["sex"] for r in records}
    folds = make_folds(subjects, sex_map, n_folds=N_FOLDS,
                       n_val=N_VAL_SUBJECTS, seed=FOLD_SEED)
    print(f"loaded {len(records)} utterances from {len(subjects)} subjects")

    audio = cache_audio(records, data_dir, raw_dir)
    records = [r for r in records if str(r["source"]) in audio]
    print(f"usable utterances with audio: {len(records)}")

    results = []
    for fold_id, fd in enumerate(folds):
        if fold >= 0 and fold_id != fold:
            continue
        tr = records_for(records, fd["train"])
        va = records_for(records, fd["val"])
        te = records_for(records, fd["test"])

        le = LabelEncoder().fit([r["syllable"] for r in tr])
        y_tr = le.transform([r["syllable"] for r in tr])
        y_va = le.transform([r["syllable"] for r in va])
        y_te = le.transform([r["syllable"] for r in te])

        wav_tr = [audio[str(r["source"])] for r in tr]
        wav_va = [audio[str(r["source"])] for r in va]
        wav_te = [audio[str(r["source"])] for r in te]

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            model_id, processor, model = load_ssl_model(model_arg, len(le.classes_))
            if processor is None:
                processor = default_processor()
            model = model.to(device)

            collate = make_collate(processor)
            tr_ld = DataLoader(AudioUttDataset(wav_tr, y_tr),
                               batch_sampler=BucketSampler([w.shape[0] for w in wav_tr], batch_size, True),
                               collate_fn=collate)
            va_ld = DataLoader(AudioUttDataset(wav_va, y_va),
                               batch_sampler=BucketSampler([w.shape[0] for w in wav_va], batch_size, False),
                               collate_fn=collate)
            te_ld = DataLoader(AudioUttDataset(wav_te, y_te),
                               batch_sampler=BucketSampler([w.shape[0] for w in wav_te], batch_size, False),
                               collate_fn=collate)

            print(f"\n[fold {fold_id} seed {seed}] train {len(tr)} / val {len(va)} / test {len(te)}")
            model, best_val = train_one_fold(model, tr_ld, va_ld, device,
                                             lr=lr, weight_decay=weight_decay,
                                             epochs=epochs, patience=patience,
                                             grad_accum=grad_accum)
            te_preds, te_truths = evaluate(model, te_ld, device, use_amp)
            m = classification_metrics(te_truths, te_preds,
                                       classes=le.classes_.tolist())
            results.append({
                "fold": fold_id, "seed": seed, "model": model_id,
                "val_acc": float(best_val),
                "test_acc": m["accuracy"], "test_f1": m["macro_f1"],
            })
            torch.save({"model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "classes": le.classes_.tolist(), "model_id": model_id},
                       os.path.join(out_dir, f"fold{fold_id}_seed{seed}_model.pth"))
            print(f"[fold {fold_id} seed {seed}] {json.dumps(results[-1])}")

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if results:
        summary = {"model": model_arg, **summarize(results)}
        summary["folds"] = results
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSSL ({model_arg}) test_acc {summary['test_acc_mean']:.4f} +- "
              f"{summary['test_acc_std']:.4f}")
    return results

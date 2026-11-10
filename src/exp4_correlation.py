"""Experiment 4: mechanistic correlation (EMA <-> acoustic).

(1) Ranks individual acoustic features by leave-one-feature-out (LOFO) accuracy
    drop on the frozen full model (a feature-level analogue of the Exp-2 group
    masking), and takes the top-10.
(2) For each top acoustic feature, computes per-utterance Pearson correlation
    against every EMA sensor feature and records the single best-correlated EMA
    feature and its |r|.
(3) For each top acoustic feature, reports the *joint* explainability of all 50
    EMA features via a global ridge regression (R^2), computed three ways so the
    "single sensor vs joint" question is answered fairly:
      - |r|^2: per-utterance single-sensor ceiling (per-utterance slope adapted),
      - ridge R^2 (1 sensor): global ridge on the best single sensor,
      - ridge R^2 (joint): global ridge on all 50 features,
      - ridge R^2 (joint, cross-subject): same but subject-CV out-of-sample.

    Both sides are within-utterance z-scored (speaker level/scale removed), so
    the comparison isolates whether the *extra* 49 sensors carry complementary
    information beyond the single best sensor. A joint R^2 substantially above
    the single-sensor R^2 means the information is jointly present in EMA (an
    architecture/optimisation limitation); a joint R^2 that stays near zero
    (F0, F3) means the information is structurally absent from the sensors.
"""
import os
import json

import numpy as np
import torch

from src.config import (ACOUSTIC_COLUMNS, EMA_COLUMNS, ALL_COLUMNS, RUNS_DIR)
from src.models.bilstm import SyllableBiLSTM
from src.exp2_shapley import eval_masked


def lofo_importance(model, seqs, labels, device, feature_cols):
    """Importance of each feature in ``feature_cols`` = accuracy drop when zeroed."""
    F = seqs[0].shape[1]
    base_acc, _ = eval_masked(model, seqs, labels, np.ones(F, bool), device)
    importances = {}
    for c in feature_cols:
        pos = ALL_COLUMNS.index(c)
        mask = np.ones(F, bool)
        mask[pos] = False
        acc, _ = eval_masked(model, seqs, labels, mask, device)
        importances[c] = base_acc - acc
    return importances, base_acc


def per_utterance_corr(records, col_index, aco_col, ema_col, voiced_only=False):
    from src.data.dataset import select
    rs = []
    for r in records:
        a = select(r, col_index, [aco_col])[:, 0].astype(np.float64)
        e = select(r, col_index, [ema_col])[:, 0].astype(np.float64)
        if voiced_only:
            voiced = select(r, col_index, ["voiced"])[:, 0] > 0.5
            a, e = a[voiced], e[voiced]
        if len(a) < 5 or a.std() < 1e-9 or e.std() < 1e-9:
            continue
        c = np.corrcoef(a, e)[0, 1]
        if np.isfinite(c):
            rs.append(c)
    return float(np.median(rs)) if rs else 0.0


def _ridge_r2(records, col_index, aco_col, x_cols, folds, voiced_only=False,
              alpha=1.0, cross_subject=False):
    """Global ridge R^2 of ``x_cols`` EMA features -> acoustic feature.

    The acoustic target is z-scored *within each utterance* (center + scale), so
    R^2 measures how well the (single or joint) EMA features track the acoustic
    feature's time course, independent of speaker level/scale. The ridge is a
    *global* model (one set of coefficients for everyone, n ~ 361k frames
    >> p features, so no overfitting). With ``cross_subject`` the ridge is fit
    per fold on the other subject-groups and evaluated on held-out subjects;
    otherwise it is fit on all frames (within-speaker physical ceiling).

    Returns {mean, median, frac_positive, n} over per-utterance R^2.
    """
    from sklearn.linear_model import Ridge
    from src.data.dataset import records_for

    x_idx = [col_index[c] for c in x_cols]
    aco_idx = col_index[aco_col]
    voiced_idx = col_index["voiced"]

    def frames(rec):
        f = rec["frame"]
        m = (f[:, voiced_idx] > 0.5) if voiced_only else np.ones(len(f), dtype=bool)
        if m.sum() < 5:
            return None, None
        x = f[m][:, x_idx].astype(np.float64)
        y = f[m][:, aco_idx].astype(np.float64)
        # within-utterance z-score BOTH sides, so the ridge models the
        # within-utterance shape tracking (speaker level/scale removed).
        ys = y.std()
        if ys < 1e-9:
            return None, None
        y = (y - y.mean()) / ys
        xs = x.std(0)
        xs[xs < 1e-9] = 1.0
        x = (x - x.mean(0, keepdims=True)) / xs
        return x, y

    def utt_r2(ridge, recs):
        r2 = []
        for r in recs:
            x, y = frames(r)
            if x is None:
                continue
            yhat = ridge.predict(x)
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float((y ** 2).sum())   # y centered
            if ss_tot > 1e-9:
                r2.append(1.0 - ss_res / ss_tot)
        return r2

    def summarize(r2):
        if not r2:
            return {"mean": 0.0, "median": 0.0, "frac_positive": 0.0, "n": 0}
        a = np.array(r2)
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "frac_positive": float((a > 0).mean()), "n": int(len(a))}

    r2_all = []
    if cross_subject:
        for fd in folds:
            tr = records_for(records, fd["train"] + fd["val"])
            te = records_for(records, fd["test"])
            Xtr, ytr = [], []
            for r in tr:
                x, y = frames(r)
                if x is not None:
                    Xtr.append(x)
                    ytr.append(y)
            if not Xtr:
                continue
            Xtr = np.concatenate(Xtr)
            ytr = np.concatenate(ytr)
            ridge = Ridge(alpha=alpha).fit(Xtr, ytr)
            r2_all.extend(utt_r2(ridge, te))
    else:
        Xall, yall = [], []
        for r in records:
            x, y = frames(r)
            if x is not None:
                Xall.append(x)
                yall.append(y)
        Xall = np.concatenate(Xall)
        yall = np.concatenate(yall)
        ridge = Ridge(alpha=alpha).fit(Xall, yall)
        r2_all.extend(utt_r2(ridge, list(records)))

    return summarize(r2_all)


def run(records, col_index, folds, full_results, out_dir=None, device="cuda",
        num_layers=2, hidden=128, dropout=0.4, top_k=10):
    device = torch.device(device)
    out_dir = out_dir or os.path.join(RUNS_DIR, "exp4")
    os.makedirs(out_dir, exist_ok=True)

    from src.data.dataset import records_for, view, fit_stats, standardize
    from src.config import GROUP_IDS as GI

    # ---- 1. LOFO importance of acoustic features (averaged over runs) ----
    imp_accum = {c: [] for c in ACOUSTIC_COLUMNS}
    for r in full_results:
        fold_id, seed = r["fold"], r["seed"]
        fd = folds[fold_id]
        tr = records_for(records, fd["train"])
        te = records_for(records, fd["test"])
        tr_seqs = view(tr, col_index, GI)
        te_seqs = view(te, col_index, GI)
        mu, sd = fit_stats(tr_seqs)
        te_std = standardize(te_seqs, mu, sd)
        ckpt = torch.load(r["ckpt_path"], map_location="cpu")
        classes = ckpt["classes"]
        cls2i = {c: i for i, c in enumerate(classes)}
        te_labels = [cls2i[rec["syllable"]] for rec in te]
        model = SyllableBiLSTM(ckpt["input_dim"], ckpt["num_classes"],
                               num_layers=num_layers, hidden=hidden, dropout=dropout)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        imp, _ = lofo_importance(model, te_std, te_labels, device, ACOUSTIC_COLUMNS)
        for c, v in imp.items():
            imp_accum[c].append(v)

    lofo_mean = {c: float(np.mean(v)) for c, v in imp_accum.items()}
    # Always include the theoretically-motivated voicing/formant features
    # (F0/F1/F2/F3) so the "structural gap" story is explicitly probed, then
    # fill the remaining slots with the top LOFO-ranked features.
    always = [c for c in ("F0", "F1", "F2", "F3") if c in lofo_mean]
    ranked = sorted(ACOUSTIC_COLUMNS, key=lambda c: lofo_mean[c], reverse=True)
    top_features = list(dict.fromkeys(always + ranked))[:top_k]
    # `voiced` is binary; a within-utterance ridge R^2 on voiced frames is
    # degenerate (constant target). It is not part of the recoverability table.
    top_features = [c for c in top_features if c != "voiced"]

    # ---- 2. correlation of top features vs all EMA features + ridge R^2 ----
    # use all utterances (correlation is a physical property, not CV-dependent);
    # the multivariate ridge R^2 uses the paper's subject CV for out-of-sample.
    voiced_set = {"F0", "voiced", "F1", "F2", "F3"}
    table = []
    for aco in top_features:
        best = None
        for ema in EMA_COLUMNS:
            r = per_utterance_corr(records, col_index, aco, ema,
                                   voiced_only=aco in voiced_set)
            if best is None or abs(r) > abs(best[1]):
                best = (ema, r)
        voiced = aco in voiced_set
        # fair "joint vs single" comparison: both global ridge, same framework
        single_in = _ridge_r2(records, col_index, aco, [best[0]], folds,
                              voiced_only=voiced, cross_subject=False)
        joint_in = _ridge_r2(records, col_index, aco, EMA_COLUMNS, folds,
                             voiced_only=voiced, cross_subject=False)
        joint_xs = _ridge_r2(records, col_index, aco, EMA_COLUMNS, folds,
                             voiced_only=voiced, cross_subject=True)
        table.append({"acoustic_feature": aco,
                      "lofo_importance": lofo_mean[aco],
                      "best_ema": best[0],
                      "best_abs_r": abs(best[1]),
                      "best_r": best[1],
                      "best_r2": float(abs(best[1]) ** 2),
                      "single_ridge_r2_in": single_in["median"],
                      "joint_ridge_r2_in": joint_in["median"],
                      "joint_ridge_r2_xs": joint_xs["median"]})

    summary = {"top_features": top_features, "lofo_importance": lofo_mean,
               "table": table}
    with open(os.path.join(out_dir, "correlation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

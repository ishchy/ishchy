#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External multi-subject validation for FFSEEG-GNN on PhysioNet EEGMMIDB.

Protocol
--------
- Dataset: PhysioNet EEG Motor Movement/Imagery Database, subjects S001-S109.
- Runs: R01 baseline eyes open (label 0), R02 baseline eyes closed (label 1).
- Common 14-channel montage: AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4.
- Sample unit: non-overlapping 2-s windows.
- Node feature: log relative alpha-band power, 8-13 Hz / 4-30 Hz.
- Evaluation: 10-fold subject-independent outer CV. No subject appears in more than
  one of train/validation/test for a fold. Validation subjects are selected only from
  the outer-training subjects. Standardization is fit on training epochs only.
- Models: MLP, FullGraphConv, GraphSAGE, FFSEEG-GNN. The FFSEEG graph, hidden size,
  K=8 and training hyperparameters reuse the frozen settings from the primary study.

The script writes fold-level, subject-level, paired-statistical, Top-K-stability and
metadata outputs. No fabricated values are used.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import ffseeg_repro_experiments as base

CHANNELS = base.CHANNELS
PHYSIONET_ROOT = "https://physionet.org/files/eegmmidb/1.0.0"
FS_EXPECTED = 160.0
WINDOW_SEC = 2.0
ALPHA_BAND = (8.0, 13.0)
TOTAL_BAND = (4.0, 30.0)


def _normalize_channel_name(s: str) -> str:
    # EDF channel labels in EEGBCI/EEGMMIDB often contain trailing periods.
    return s.upper().replace(".", "").replace(" ", "")


def _download(url: str, out: Path, retries: int = 4) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 1024:
        return
    last = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url} (attempt {attempt}/{retries})", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 FFSEEG-research"})
            with urllib.request.urlopen(req, timeout=90) as r, open(out, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if out.stat().st_size <= 1024:
                raise IOError(f"Downloaded file is unexpectedly small: {out}")
            return
        except Exception as exc:
            last = exc
            if out.exists():
                out.unlink(missing_ok=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last}")


def _bandpower_feature(data: np.ndarray, sfreq: float, window_sec: float = WINDOW_SEC) -> np.ndarray:
    """Return [n_windows, n_channels] log relative alpha power.

    Power is computed by a Hann-windowed one-sided FFT within each non-overlapping
    window. Relative alpha power = P(8-13 Hz) / P(4-30 Hz), then log transform.
    """
    n_ch, n_time = data.shape
    n_win = int(round(window_sec * sfreq))
    n_windows = n_time // n_win
    if n_windows < 1:
        raise ValueError("Recording shorter than one analysis window.")
    x = data[:, : n_windows * n_win].reshape(n_ch, n_windows, n_win).transpose(1, 0, 2)
    x = x - x.mean(axis=-1, keepdims=True)
    w = np.hanning(n_win).astype(np.float64)
    spec = np.fft.rfft(x * w[None, None, :], axis=-1)
    psd = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n_win, d=1.0 / sfreq)
    am = (freqs >= ALPHA_BAND[0]) & (freqs <= ALPHA_BAND[1])
    tm = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    alpha = psd[..., am].sum(axis=-1)
    total = psd[..., tm].sum(axis=-1)
    rel = (alpha + 1e-18) / (total + 1e-18)
    return np.log(rel + 1e-12).astype(np.float32)


def extract_real_dataset(cache_dir: Path, subjects: Iterable[int]) -> Tuple[pd.DataFrame, Dict]:
    import mne

    rows = []
    audit = []
    for si, subject in enumerate(subjects, 1):
        subj = f"S{subject:03d}"
        per_subject = []
        for run, label in [(1, 0), (2, 1)]:
            name = f"{subj}R{run:02d}.edf"
            url = f"{PHYSIONET_ROOT}/{subj}/{name}"
            path = cache_dir / subj / name
            _download(url, path)
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            sfreq = float(raw.info["sfreq"])
            if not math.isclose(sfreq, FS_EXPECTED, rel_tol=0, abs_tol=1e-6):
                raise RuntimeError(f"Unexpected sampling rate for {name}: {sfreq}")
            norm_to_orig = {_normalize_channel_name(c): c for c in raw.ch_names}
            missing = [c for c in CHANNELS if c.upper() not in norm_to_orig]
            if missing:
                raise RuntimeError(f"Missing required channels in {name}: {missing}; raw={raw.ch_names}")
            picks = [norm_to_orig[c.upper()] for c in CHANNELS]
            data = raw.get_data(picks=picks)
            feat = _bandpower_feature(data, sfreq)
            for wi, vec in enumerate(feat):
                rec = {ch: float(vec[j]) for j, ch in enumerate(CHANNELS)}
                rec.update({"subject": subject, "run": run, "window": wi, "label": label})
                rows.append(rec)
            per_subject.append(len(feat))
            audit.append({"subject": subject, "run": run, "label": label, "sfreq": sfreq,
                          "n_samples": int(data.shape[1]), "n_windows": int(len(feat)),
                          "channels_ok": True})
        print(f"Extracted {subj} ({si}): open={per_subject[0]} closed={per_subject[1]} windows", flush=True)
    df = pd.DataFrame(rows)
    meta = {
        "n_subjects": int(df["subject"].nunique()),
        "subjects": sorted(int(x) for x in df["subject"].unique()),
        "n_epochs": int(len(df)),
        "class_counts": {str(int(k)): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "channels": CHANNELS,
        "feature": "log relative alpha power = log(P[8,13] Hz / P[4,30] Hz)",
        "window_sec": WINDOW_SEC,
        "window_overlap": 0.0,
        "sampling_rate_expected": FS_EXPECTED,
        "audit": audit,
    }
    return df, meta


def synthetic_dataset(n_subjects: int = 20, epochs_per_class: int = 8, seed: int = 7) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(1, n_subjects + 1):
        subj_shift = rng.normal(0, 0.4, size=len(CHANNELS))
        for label in [0, 1]:
            for w in range(epochs_per_class):
                # Mild class signal on posterior channels; smoke-test only.
                x = subj_shift + rng.normal(0, 0.8, size=len(CHANNELS))
                if label == 1:
                    for ch in ["O1", "O2", "P7", "P8"]:
                        x[CHANNELS.index(ch)] += 1.0
                rec = {ch: float(x[j]) for j, ch in enumerate(CHANNELS)}
                rec.update({"subject": s, "run": label + 1, "window": w, "label": label})
                rows.append(rec)
    df = pd.DataFrame(rows)
    return df, {"synthetic_smoke": True, "n_subjects": n_subjects, "n_epochs": len(df)}


def metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    pred = logits.argmax(1)
    prob = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "auc": float(roc_auc_score(y_true, prob)),
    }


def train_model(name: str, A, A_norm, A_mean, split, cfg, device):
    Xtr, ytr, Xva, yva, Xte, yte = split
    aux = {}
    if name == "MLP":
        model = base.MLP(dropout=0.10).to(device)
        model = base._train_generic(model, Xtr, ytr, Xva, yva, cfg, device, lr=cfg.lr_default)
    elif name == "FullGraphConv":
        model = base.FullGraphConv(A_norm, dropout=0.10).to(device)
        model = base._train_generic(model, Xtr, ytr, Xva, yva, cfg, device, lr=cfg.lr_fullgcn)
    elif name == "GraphSAGE":
        model = base.GraphSAGE(A_mean, dropout=0.10).to(device)
        model = base._train_generic(model, Xtr, ytr, Xva, yva, cfg, device, lr=cfg.lr_default)
    elif name == "FFSEEG-GNN":
        model, _, aux = base.train_ffseeg(A_norm, Xtr, ytr, Xva, yva, cfg, device)
    else:
        raise KeyError(name)
    logits = base.infer_logits(model, Xte, cfg.batch_size, device)
    return model, logits, metrics(yte, logits), aux


def make_subject_folds(df: pd.DataFrame, n_splits: int, split_seed: int):
    subjects = np.array(sorted(df["subject"].unique()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
    for fold, (tr_sub_idx, te_sub_idx) in enumerate(kf.split(subjects), 1):
        outer_train_subjects = subjects[tr_sub_idx]
        test_subjects = subjects[te_sub_idx]
        # Validation is subject-wise and selected only from outer-training subjects.
        rng = np.random.default_rng(split_seed + 1000 + fold)
        shuffled = outer_train_subjects.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(round(0.15 * len(shuffled))))
        val_subjects = np.sort(shuffled[:n_val])
        train_subjects = np.sort(shuffled[n_val:])
        yield fold, train_subjects, val_subjects, np.sort(test_subjects)


def paired_stats(fold_df: pd.DataFrame, ref: str = "FFSEEG-GNN") -> pd.DataFrame:
    rows = []
    metric = "balanced_accuracy"
    pvals = []
    temp = []
    ref_map = fold_df[fold_df.model == ref].set_index("fold")[metric]
    for model in sorted(m for m in fold_df.model.unique() if m != ref):
        other = fold_df[fold_df.model == model].set_index("fold")[metric]
        common = ref_map.index.intersection(other.index)
        d = (ref_map.loc[common] - other.loc[common]).to_numpy(float)
        try:
            stat = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
            p = float(stat.pvalue)
        except Exception:
            p = np.nan
        # paired rank-biserial-like sign consistency summary; not population inference.
        effect = float((np.sum(d > 0) - np.sum(d < 0)) / len(d)) if len(d) else np.nan
        temp.append((model, d, p, effect))
        pvals.append(p)
    # Holm correction
    finite_idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    holm = [np.nan] * len(pvals)
    if finite_idx:
        order = sorted(finite_idx, key=lambda i: pvals[i])
        running = 0.0
        m = len(order)
        for rank, i in enumerate(order):
            adj = min(1.0, (m - rank) * pvals[i])
            running = max(running, adj)
            holm[i] = running
    for i, (model, d, p, effect) in enumerate(temp):
        rows.append({
            "reference": ref, "comparison": model, "metric": metric,
            "mean_difference_pp": float(d.mean() * 100),
            "std_difference_pp": float(d.std(ddof=1) * 100),
            "wilcoxon_p": p, "holm_p": holm[i], "sign_effect": effect,
            "n_folds": int(len(d)),
        })
    return pd.DataFrame(rows)


def run_cv(df: pd.DataFrame, outdir: Path, n_splits: int, split_seed: int, max_epochs: int, batch_size: int):
    device = torch.device("cpu")
    cfg = base.TrainConfig(batch_size=batch_size, max_epochs=max_epochs, k=8)
    A, A_norm, A_mean = base.build_adjacency(device)
    X_all = df[CHANNELS].to_numpy(np.float32)
    y_all = df["label"].to_numpy(np.int64)
    subjects_all = df["subject"].to_numpy(np.int64)
    models = ["MLP", "FullGraphConv", "GraphSAGE", "FFSEEG-GNN"]

    fold_rows: List[Dict] = []
    prediction_rows: List[Dict] = []
    topk_rows: List[Dict] = []
    split_rows: List[Dict] = []

    for fold, tr_sub, va_sub, te_sub in make_subject_folds(df, n_splits, split_seed):
        tr = np.isin(subjects_all, tr_sub)
        va = np.isin(subjects_all, va_sub)
        te = np.isin(subjects_all, te_sub)
        scaler = StandardScaler().fit(X_all[tr])
        Xtr = scaler.transform(X_all[tr]).astype(np.float32)
        Xva = scaler.transform(X_all[va]).astype(np.float32)
        Xte = scaler.transform(X_all[te]).astype(np.float32)
        ytr, yva, yte = y_all[tr], y_all[va], y_all[te]
        split_rows.append({
            "fold": fold,
            "train_subjects": ",".join(map(str, tr_sub.tolist())),
            "validation_subjects": ",".join(map(str, va_sub.tolist())),
            "test_subjects": ",".join(map(str, te_sub.tolist())),
            "n_train_subjects": len(tr_sub), "n_val_subjects": len(va_sub), "n_test_subjects": len(te_sub),
            "n_train_epochs": int(tr.sum()), "n_val_epochs": int(va.sum()), "n_test_epochs": int(te.sum()),
            "train_pos_rate": float(ytr.mean()), "val_pos_rate": float(yva.mean()), "test_pos_rate": float(yte.mean()),
        })
        print(f"Fold {fold}/{n_splits}: train={len(tr_sub)} val={len(va_sub)} test={len(te_sub)} subjects", flush=True)
        split = (Xtr, ytr, Xva, yva, Xte, yte)
        test_indices = np.where(te)[0]
        for mi, name in enumerate(models):
            seed = 12000 + fold * 100 + mi
            base.seed_everything(seed)
            print(f"  Training {name} seed={seed}", flush=True)
            _, logits, m, aux = train_model(name, A, A_norm, A_mean, split, cfg, device)
            fold_rows.append({"fold": fold, "seed": seed, "model": name, **m})
            prob = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
            pred = logits.argmax(1)
            for local_i, global_i in enumerate(test_indices):
                prediction_rows.append({
                    "fold": fold, "model": name,
                    "subject": int(subjects_all[global_i]),
                    "run": int(df.iloc[global_i]["run"]),
                    "window": int(df.iloc[global_i]["window"]),
                    "y_true": int(y_all[global_i]), "y_pred": int(pred[local_i]),
                    "prob_closed": float(prob[local_i]),
                })
            if name == "FFSEEG-GNN":
                topk_rows.append({
                    "fold": fold,
                    "top_channels": ",".join(aux.get("top_channels", [])),
                    "top_indices": ",".join(map(str, aux.get("top_indices", []))),
                    "soft_best_val_f1": aux.get("soft_best_val_f1", np.nan),
                })

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    topk_df = pd.DataFrame(topk_rows)
    split_df = pd.DataFrame(split_rows)
    fold_df.to_csv(outdir / "external_fold_metrics.csv", index=False)
    pred_df.to_csv(outdir / "external_oof_predictions.csv", index=False)
    topk_df.to_csv(outdir / "external_ffseeg_topk_by_fold.csv", index=False)
    split_df.to_csv(outdir / "external_subject_splits.csv", index=False)

    # Fold-level summary
    metric_cols = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.to_csv(outdir / "external_fold_summary.csv")

    # Subject-level metrics from out-of-fold predictions.
    subj_rows = []
    for (model, subject), g in pred_df.groupby(["model", "subject"]):
        yt = g.y_true.to_numpy(int)
        yp = g.y_pred.to_numpy(int)
        pr = g.prob_closed.to_numpy(float)
        subj_rows.append({
            "model": model, "subject": int(subject),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
            "auc": roc_auc_score(yt, pr),
            "n_epochs": len(g),
        })
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(outdir / "external_subject_metrics.csv", index=False)
    subj_summary = subj_df.groupby("model")[metric_cols].agg(["mean", "std", "median"])
    subj_summary.to_csv(outdir / "external_subject_summary.csv")

    paired_stats(fold_df).to_csv(outdir / "external_fold_paired_stats.csv", index=False)

    # Top-K stability across outer folds.
    sets = []
    freq = {ch: 0 for ch in CHANNELS}
    for s in topk_df.top_channels:
        st = set(x for x in s.split(",") if x)
        sets.append(st)
        for ch in st:
            freq[ch] += 1
    jac = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = sets[i] | sets[j]
            jac.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
    stab = {
        "n_folds": len(sets),
        "mean_pairwise_jaccard": float(np.mean(jac)) if jac else np.nan,
        "std_pairwise_jaccard": float(np.std(jac, ddof=1)) if len(jac) > 1 else np.nan,
        "selection_frequency": freq,
    }
    (outdir / "external_topk_stability.json").write_text(json.dumps(stab, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="ffseeg_physionet_external")
    ap.add_argument("--cache-dir", default="physionet_eegmmidb_cache")
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--split-seed", type=int, default=20260918)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--subject-start", type=int, default=1)
    ap.add_argument("--subject-end", type=int, default=109)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        df, dataset_meta = synthetic_dataset(n_subjects=max(args.n_splits, 20), epochs_per_class=6)
    else:
        subjects = range(args.subject_start, args.subject_end + 1)
        df, dataset_meta = extract_real_dataset(Path(args.cache_dir), subjects)
    df.to_csv(outdir / "external_features.csv", index=False)
    cfg = run_cv(df, outdir, args.n_splits, args.split_seed, args.max_epochs, args.batch_size)
    meta = {
        "dataset": "PhysioNet EEG Motor Movement/Imagery Database v1.0.0",
        "dataset_url": "https://physionet.org/content/eegmmidb/1.0.0/",
        "task": "baseline eyes open (R01) vs eyes closed (R02)",
        "evaluation": f"{args.n_splits}-fold subject-independent outer cross-validation",
        "split_seed": args.split_seed,
        "training_config": asdict(cfg),
        "dataset_meta": dataset_meta,
        "models": ["MLP", "FullGraphConv", "GraphSAGE", "FFSEEG-GNN"],
        "frozen_primary_study_settings": {"graph_edges": base.EDGES, "K": 8, "channels": CHANNELS},
        "smoke": args.smoke,
    }
    (outdir / "external_run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("External validation finished:", outdir.resolve(), flush=True)


if __name__ == "__main__":
    main()
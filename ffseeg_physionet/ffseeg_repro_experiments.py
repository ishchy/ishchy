#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FFSEEG-GNN supplementary experiments (reproducible, no fabricated results)

What this script does
---------------------
1) Downloads/loads the public UCI EEG Eye State data.
2) Removes the four widely reported outliers (1-based rows 899, 10387, 11510, 13180)
   unless --keep-outliers is used.
3) Implements the fixed 14-electrode graph described in the manuscript.
4) Implements the following models without torch_geometric:
   - MLP
   - FullGraphConv
   - FFSEEG-GNN (soft -> fixed global Top-K -> hard fine-tuning)
   - GraphSAGE
   - GAT
   - GCN-TopKPool (per-sample Top-K)
   - GCN-SAGPool (per-sample structure-aware Top-K)
5) Runs:
   - repeated random stratified evaluation
   - time-aware grouped holdout with purge buffer
   - strong graph baseline comparison
   - outlier sensitivity
   - CPU latency / parameter-count benchmark
6) Saves CSV summaries and JSON metadata in the selected output directory.

IMPORTANT
---------
The --smoke-test mode uses synthetic data ONLY to verify that the code executes.
Synthetic smoke-test results must NOT be reported as empirical results in a paper.

The original training code for the manuscript was unavailable, so exact numerical
reproduction of the manuscript's existing tables cannot be guaranteed. If you use
this script for a submission, use its results consistently across all affected tables
and do not mix incompatible old/new runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- reproducibility ----------------------------- #

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is useful for paired model comparisons. Some GPU kernels can
    # still differ slightly across hardware/PyTorch versions.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# ----------------------------- data loading -------------------------------- #

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
OUTLIER_ROWS_1BASED = [899, 10387, 11510, 13180]
DOWNLOAD_URLS = [
    "https://raw.githubusercontent.com/datasets/eeg-eye-state/main/data/eeg-eye-state.csv",
    "https://datahub.io/core/eeg-eye-state/_r/-/data/eeg-eye-state.csv",
]


def _read_local_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".arff":
        raw, _meta = arff.loadarff(path)
        df = pd.DataFrame(raw)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}; use CSV or ARFF")
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    # common mirrors use P instead of P7 and class instead of eyeDetection
    if "P" in df.columns and "P7" not in df.columns:
        rename["P"] = "P7"
    if "class" in df.columns and "eyeDetection" not in df.columns:
        rename["class"] = "eyeDetection"
    if "eyeDetection" not in df.columns:
        # ARFF target may be named eyeDetection with bytes; preserve if present.
        candidates = [c for c in df.columns if str(c).lower() in {"class", "target", "label", "eyedetection"}]
        if candidates:
            rename[candidates[-1]] = "eyeDetection"
    df = df.rename(columns=rename)
    missing = [c for c in CHANNELS + ["eyeDetection"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}; found {list(df.columns)}")
    df = df[CHANNELS + ["eyeDetection"]].copy()
    for c in CHANNELS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["eyeDetection"] = pd.to_numeric(df["eyeDetection"], errors="raise").astype(int)
    return df


def download_dataset(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in DOWNLOAD_URLS:
        try:
            print(f"Downloading dataset from: {url}")
            urllib.request.urlretrieve(url, destination)
            if destination.exists() and destination.stat().st_size > 100_000:
                return destination
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(
        "Automatic download failed. Download the public EEG Eye State CSV/ARFF manually "
        "and pass it with --data PATH.\n" + "\n".join(errors)
    )


def make_smoke_data(n: int = 1600, seed: int = 7) -> pd.DataFrame:
    """Synthetic data for code-path verification only; never for paper results."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    # spatially correlated synthetic channels; not meant to mimic physiology
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    X = np.empty((n, 14), dtype=np.float32)
    for j in range(14):
        X[:, j] = 4300 + 18 * z1 + 7 * z2 + y * (0.8 + 0.15 * (j % 4)) + rng.normal(0, 7, size=n)
    df = pd.DataFrame(X, columns=CHANNELS)
    df["eyeDetection"] = y
    return df


def load_eeg(data_path: Optional[str], keep_outliers: bool, smoke_test: bool, cache_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    if smoke_test:
        df = make_smoke_data()
        return df, {"source": "SYNTHETIC_SMOKE_TEST", "outliers_removed": []}

    if data_path:
        path = Path(data_path)
    else:
        path = cache_dir / "eeg-eye-state.csv"
        if not path.exists():
            path = download_dataset(path)

    df = _normalize_columns(_read_local_data(path))
    meta = {"source": str(path), "raw_rows": int(len(df))}
    if len(df) != 14980:
        print(f"WARNING: expected 14980 rows, found {len(df)}. Check source version.")

    removed = []
    if not keep_outliers:
        idx0 = [i - 1 for i in OUTLIER_ROWS_1BASED if 1 <= i <= len(df)]
        removed = OUTLIER_ROWS_1BASED.copy()
        df = df.drop(index=idx0).reset_index(drop=True)
    meta["outliers_removed"] = removed
    meta["rows_after_cleaning"] = int(len(df))
    meta["class_counts"] = {str(k): int(v) for k, v in df["eyeDetection"].value_counts().sort_index().items()}
    return df, meta


# ----------------------------- graph --------------------------------------- #

EDGES = [
    ("AF3", "F7"), ("AF3", "F3"), ("AF3", "AF4"),
    ("F7", "F3"), ("F7", "FC5"), ("F7", "T7"),
    ("F3", "FC5"), ("F3", "F4"),
    ("FC5", "T7"), ("FC5", "P7"), ("FC5", "FC6"),
    ("T7", "P7"), ("T7", "T8"),
    ("P7", "O1"), ("P7", "P8"),
    ("O1", "O2"), ("O2", "P8"),
    ("P8", "T8"),
    ("T8", "FC6"), ("T8", "F8"),
    ("FC6", "F4"), ("FC6", "F8"),
    ("F4", "F8"), ("F4", "AF4"), ("F8", "AF4"),
]


def build_adjacency(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(CHANNELS)
    idx = {c: i for i, c in enumerate(CHANNELS)}
    A = torch.zeros((n, n), dtype=torch.float32)
    for a, b in EDGES:
        i, j = idx[a], idx[b]
        A[i, j] = 1.0
        A[j, i] = 1.0
    A_no_self = A.clone()
    A_self = A + torch.eye(n)
    deg = A_self.sum(1)
    Dinv = torch.diag(torch.pow(deg.clamp_min(1.0), -0.5))
    A_norm = Dinv @ A_self @ Dinv
    # row-normalized no-self matrix for GraphSAGE mean aggregation
    deg_ns = A_no_self.sum(1, keepdim=True).clamp_min(1.0)
    A_mean = A_no_self / deg_ns
    return A.to(device), A_norm.to(device), A_mean.to(device)


# ----------------------------- models -------------------------------------- #

class DenseHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return self.out(x)


class MLP(nn.Module):
    def __init__(self, n_nodes: int = 14, dropout: float = 0.10):
        super().__init__()
        self.head = DenseHead(n_nodes, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x.squeeze(-1))


class TwoBranchGraphConv(nn.Module):
    """H' = H W_self + A_norm H W_neigh + b."""
    def __init__(self, in_dim: int, out_dim: int, A_norm: torch.Tensor):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.register_buffer("A_norm", A_norm)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjf->bif", self.A_norm, h)
        return self.self_lin(h) + self.neigh_lin(neigh) + self.bias


class GCNEncoder(nn.Module):
    def __init__(self, A_norm: torch.Tensor, hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.g1 = TwoBranchGraphConv(1, hidden, A_norm)
        self.g2 = TwoBranchGraphConv(hidden, hidden, A_norm)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.g1(x)))
        h = F.relu(self.g2(h))
        return h


class FullGraphConv(nn.Module):
    def __init__(self, A_norm: torch.Tensor, hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.encoder = GCNEncoder(A_norm, hidden, dropout)
        self.head = DenseHead(14 * hidden, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h.flatten(1))


class FFSEEGSoft(nn.Module):
    def __init__(self, A_norm: torch.Tensor, hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.encoder = GCNEncoder(A_norm, hidden, dropout)
        self.score_vec = nn.Parameter(torch.empty(hidden))
        self.score_bias = nn.Parameter(torch.zeros(14))
        nn.init.normal_(self.score_vec, std=0.10)
        self.head = DenseHead(14 * hidden, dropout)
    def forward(self, x: torch.Tensor, tau: float = 0.9):
        h = self.encoder(x)
        logits = torch.einsum("bnh,h->bn", h, self.score_vec) + self.score_bias
        gates = torch.sigmoid(logits / max(float(tau), 1e-4))
        h_g = h * gates.unsqueeze(-1)
        out = self.head(h_g.flatten(1))
        return out, gates, logits, h


class FFSEEGHard(nn.Module):
    def __init__(self, encoder: GCNEncoder, top_idx: List[int], hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("top_idx", torch.tensor(top_idx, dtype=torch.long))
        self.head = DenseHead(len(top_idx) * hidden, dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        hs = h.index_select(1, self.top_idx)
        return self.head(hs.flatten(1))


class SAGEConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, A_mean: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.register_buffer("A_mean", A_mean)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjf->bif", self.A_mean, h)
        return self.lin(torch.cat([h, neigh], dim=-1))


class GraphSAGE(nn.Module):
    def __init__(self, A_mean: torch.Tensor, hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.s1 = SAGEConv(1, hidden, A_mean)
        self.s2 = SAGEConv(hidden, hidden, A_mean)
        self.dropout = nn.Dropout(dropout)
        self.head = DenseHead(14 * hidden, dropout)
    def forward(self, x):
        h = self.dropout(F.relu(self.s1(x)))
        h = F.relu(self.s2(h))
        return self.head(h.flatten(1))


class DenseGATLayer(nn.Module):
    """Small dense masked GAT layer suitable for N=14 fixed graph."""
    def __init__(self, in_dim: int, out_dim: int, adj_with_self: torch.Tensor, heads: int = 1, concat: bool = True, dropout: float = 0.10):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Parameter(torch.empty(heads, in_dim, out_dim))
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", adj_with_self > 0)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # z: [B,H,N,F]
        z = torch.einsum("bni,hio->bhno", x, self.W)
        e_src = torch.einsum("bhno,ho->bhn", z, self.a_src)
        e_dst = torch.einsum("bhno,ho->bhn", z, self.a_dst)
        e = F.leaky_relu(e_src.unsqueeze(-1) + e_dst.unsqueeze(-2), negative_slope=0.2)
        mask = self.mask.unsqueeze(0).unsqueeze(0)
        e = e.masked_fill(~mask, -1e9)
        alpha = self.dropout(torch.softmax(e, dim=-1))
        out = torch.einsum("bhij,bhjo->bhio", alpha, z)
        if self.concat:
            out = out.permute(0, 2, 1, 3).contiguous().view(x.shape[0], x.shape[1], self.heads * self.out_dim)
        else:
            out = out.mean(1)
        return out


class GAT(nn.Module):
    def __init__(self, A: torch.Tensor, dropout: float = 0.10):
        super().__init__()
        Aself = A + torch.eye(A.shape[0], device=A.device)
        self.g1 = DenseGATLayer(1, 8, Aself, heads=2, concat=True, dropout=dropout)  # 16 dims
        self.g2 = DenseGATLayer(16, 16, Aself, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.head = DenseHead(14 * 16, dropout)
    def forward(self, x):
        h = self.dropout(F.elu(self.g1(x)))
        h = F.elu(self.g2(h))
        return self.head(h.flatten(1))


class GCNTopKPool(nn.Module):
    """GCN encoder + per-sample projection Top-K pooling."""
    def __init__(self, A_norm: torch.Tensor, k: int = 8, hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.encoder = GCNEncoder(A_norm, hidden, dropout)
        self.proj = nn.Parameter(torch.empty(hidden))
        nn.init.normal_(self.proj, std=0.10)
        self.k = k
        self.head = DenseHead(k * hidden, dropout)
    def forward(self, x):
        h = self.encoder(x)
        p = self.proj / (self.proj.norm() + 1e-8)
        scores = torch.einsum("bnh,h->bn", h, p)
        vals, idx = torch.topk(scores, self.k, dim=1, largest=True, sorted=True)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        hs = torch.gather(h, 1, gather_idx) * torch.tanh(vals).unsqueeze(-1)
        return self.head(hs.flatten(1))


class SAGScore(nn.Module):
    def __init__(self, A_norm: torch.Tensor, in_dim: int):
        super().__init__()
        self.g = TwoBranchGraphConv(in_dim, 1, A_norm)
    def forward(self, h):
        return self.g(h).squeeze(-1)


class GCNSAGPool(nn.Module):
    """GCN encoder + graph-convolutional node scoring + per-sample Top-K."""
    def __init__(self, A_norm: torch.Tensor, k: int = 8, hidden: int = 16, dropout: float = 0.10):
        super().__init__()
        self.encoder = GCNEncoder(A_norm, hidden, dropout)
        self.score = SAGScore(A_norm, hidden)
        self.k = k
        self.head = DenseHead(k * hidden, dropout)
    def forward(self, x):
        h = self.encoder(x)
        scores = self.score(h)
        vals, idx = torch.topk(scores, self.k, dim=1, largest=True, sorted=True)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, h.shape[-1])
        hs = torch.gather(h, 1, gather_idx) * torch.tanh(vals).unsqueeze(-1)
        return self.head(hs.flatten(1))


# ----------------------------- training utilities -------------------------- #

@dataclass
class TrainConfig:
    batch_size: int = 256
    max_epochs: int = 120
    patience: int = 14
    check_every: int = 5
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    lr_default: float = 0.008
    lr_fullgcn: float = 0.010
    lr_soft: float = 0.008
    lr_hard: float = 0.007
    lambda_k: float = 0.025
    lambda_disc: float = 0.003
    tau0: float = 0.90
    tau_decay: float = 0.985
    tau_min: float = 0.30
    k: int = 8


def make_loader(X, y, batch_size, shuffle, device=None):
    xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    yt = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    prob = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    pred = logits.argmax(1)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, prob),
    }


def infer_logits(model, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = make_loader(X, np.zeros(len(X), dtype=np.int64), batch_size, False)
    outs = []
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device)
            out = model(xb)
            if isinstance(out, tuple):
                out = out[0]
            outs.append(out.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _train_generic(model: nn.Module, Xtr, ytr, Xva, yva, cfg: TrainConfig, device: torch.device, lr: float) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    loader = make_loader(Xtr, ytr, cfg.batch_size, True)
    best_state = None
    best_f1 = -np.inf
    bad_checks = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        if epoch % cfg.check_every == 0:
            va_logits = infer_logits(model, Xva, cfg.batch_size, device)
            score = f1_score(yva, va_logits.argmax(1), zero_division=0)
            if score > best_f1 + 1e-6:
                best_f1 = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_checks = 0
            else:
                bad_checks += 1
                if bad_checks >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_ffseeg(A_norm, Xtr, ytr, Xva, yva, cfg: TrainConfig, device: torch.device) -> Tuple[nn.Module, List[int], Dict]:
    soft = FFSEEGSoft(A_norm, hidden=16, dropout=0.10).to(device)
    opt = torch.optim.AdamW(soft.parameters(), lr=cfg.lr_soft, weight_decay=cfg.weight_decay)
    loader = make_loader(Xtr, ytr, cfg.batch_size, True)
    best_state = None
    best_f1 = -np.inf
    bad_checks = 0
    tau = cfg.tau0

    for epoch in range(1, cfg.max_epochs + 1):
        soft.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, gates, _, _ = soft(xb, tau=tau)
            ce = F.cross_entropy(logits, yb)
            # Mean retained-node count should approach K.
            k_loss = ((gates.sum(dim=1).mean() - cfg.k) / 14.0) ** 2
            # Penalize ambiguous 0.5 gates; minimum is at 0 or 1.
            disc = (gates * (1.0 - gates)).mean()
            loss = ce + cfg.lambda_k * k_loss + cfg.lambda_disc * disc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(soft.parameters(), cfg.grad_clip)
            opt.step()
        tau = max(cfg.tau_min, tau * cfg.tau_decay)
        if epoch % cfg.check_every == 0:
            # validation uses current soft model only to select best encoder/scoring state
            soft.eval()
            outs = []
            with torch.inference_mode():
                for xb, _ in make_loader(Xva, yva, cfg.batch_size, False):
                    xb = xb.to(device)
                    out, _, _, _ = soft(xb, tau=tau)
                    outs.append(out.cpu().numpy())
            va_logits = np.concatenate(outs)
            score = f1_score(yva, va_logits.argmax(1), zero_division=0)
            if score > best_f1 + 1e-6:
                best_f1 = score
                best_state = {k: v.detach().cpu().clone() for k, v in soft.state_dict().items()}
                bad_checks = 0
            else:
                bad_checks += 1
                if bad_checks >= cfg.patience:
                    break
    if best_state is not None:
        soft.load_state_dict(best_state)

    # Training-set mean score only; no validation/test information is used for Top-K index fixing.
    soft.eval()
    score_sum = torch.zeros(14, device=device)
    n_seen = 0
    with torch.inference_mode():
        for xb, _ in make_loader(Xtr, ytr, cfg.batch_size, False):
            xb = xb.to(device)
            _, _, raw_score, _ = soft(xb, tau=cfg.tau_min)
            score_sum += raw_score.sum(dim=0)
            n_seen += xb.shape[0]
    mean_score = score_sum / max(n_seen, 1)
    top_idx = torch.topk(mean_score, cfg.k, largest=True, sorted=True).indices.cpu().tolist()

    # Copy the encoder to hard model; use a new classifier because input dimension changes.
    hard = FFSEEGHard(soft.encoder, top_idx, hidden=16, dropout=0.10).to(device)
    hard = _train_generic(hard, Xtr, ytr, Xva, yva, cfg, device, lr=cfg.lr_hard)
    aux = {
        "top_indices": top_idx,
        "top_channels": [CHANNELS[i] for i in top_idx],
        "soft_best_val_f1": float(best_f1),
        "mean_scores": mean_score.detach().cpu().numpy().tolist(),
    }
    return hard, top_idx, aux


# ----------------------------- splitting ----------------------------------- #

def standardize_split(Xtr, Xva, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr).astype(np.float32), sc.transform(Xva).astype(np.float32), sc.transform(Xte).astype(np.float32), sc


def random_stratified_split(X: np.ndarray, y: np.ndarray, seed: int):
    Xdev, Xte, ydev, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    # 15% of the 75% development subset -> 11.25% of the full data.
    Xtr, Xva, ytr, yva = train_test_split(Xdev, ydev, test_size=0.15, stratify=ydev, random_state=seed)
    Xtr, Xva, Xte, _ = standardize_split(Xtr, Xva, Xte)
    return Xtr, ytr, Xva, yva, Xte, yte


def _best_group_holdout(groups: np.ndarray, y: np.ndarray, fraction: float, seed: int, n_candidates: int = 300) -> np.ndarray:
    """Choose group IDs by a fixed label-balance/size criterion; never by model performance."""
    rng = np.random.default_rng(seed)
    gids = np.unique(groups)
    n_select = max(1, int(round(len(gids) * fraction)))
    global_p = y.mean()
    target_n = len(y) * fraction
    best = None
    best_score = np.inf
    for _ in range(n_candidates):
        sel = rng.choice(gids, size=n_select, replace=False)
        mask = np.isin(groups, sel)
        if mask.sum() == 0 or mask.sum() == len(y):
            continue
        p = y[mask].mean()
        size_pen = abs(mask.sum() - target_n) / len(y)
        class_pen = abs(p - global_p)
        score = class_pen + 0.25 * size_pen
        if score < best_score:
            best_score = score
            best = np.sort(sel)
    if best is None:
        raise RuntimeError("Failed to construct grouped holdout.")
    return best


def grouped_time_split(X: np.ndarray, y: np.ndarray, seed: int, group_size: int = 256, purge_groups: int = 1):
    """
    Time-aware grouped holdout.

    - Consecutive samples are packed into fixed contiguous groups (~2 s for 128 Hz data).
    - Test groups are selected as whole groups with a fixed class-balance criterion.
    - Immediate neighboring groups are purged from training to reduce boundary leakage.
    - Validation is selected from remaining development groups in the same way.

    This is intentionally less extreme than one entire contiguous fifth as test, while
    still preventing sample-level random leakage.
    """
    n = len(y)
    groups = np.arange(n) // group_size
    test_g = _best_group_holdout(groups, y, fraction=0.20, seed=seed)
    test_mask = np.isin(groups, test_g)

    purge_test = set()
    for g in test_g:
        for d in range(-purge_groups, purge_groups + 1):
            purge_test.add(int(g + d))
    dev_mask = ~np.isin(groups, list(purge_test))

    dev_idx = np.where(dev_mask)[0]
    dev_groups = groups[dev_idx]
    dev_y = y[dev_idx]
    # Validation ~15% of original development region; choose 15% of dev groups.
    val_g = _best_group_holdout(dev_groups, dev_y, fraction=0.15, seed=seed + 10000)
    val_mask_full = np.zeros(n, dtype=bool)
    val_mask_full[dev_idx[np.isin(dev_groups, val_g)]] = True

    purge_val = set()
    for g in val_g:
        for d in range(-purge_groups, purge_groups + 1):
            purge_val.add(int(g + d))
    train_mask = dev_mask & ~np.isin(groups, list(purge_val)) & ~val_mask_full

    tr, va, te = np.where(train_mask)[0], np.where(val_mask_full)[0], np.where(test_mask)[0]
    if min(len(tr), len(va), len(te)) == 0:
        raise RuntimeError("Grouped split created an empty partition; increase data or reduce purge.")
    Xtr, Xva, Xte, _ = standardize_split(X[tr], X[va], X[te])
    return Xtr, y[tr], Xva, y[va], Xte, y[te], {
        "group_size": group_size, "purge_groups": purge_groups,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "train_pos_rate": float(y[tr].mean()), "val_pos_rate": float(y[va].mean()), "test_pos_rate": float(y[te].mean())
    }


# ----------------------------- experiment runner --------------------------- #

def model_factory(name: str, A, A_norm, A_mean, cfg: TrainConfig):
    if name == "MLP":
        return MLP()
    if name == "FullGraphConv":
        return FullGraphConv(A_norm)
    if name == "GraphSAGE":
        return GraphSAGE(A_mean)
    if name == "GAT":
        return GAT(A)
    if name == "GCN-TopKPool":
        return GCNTopKPool(A_norm, k=cfg.k)
    if name == "GCN-SAGPool":
        return GCNSAGPool(A_norm, k=cfg.k)
    raise KeyError(name)


def train_and_eval(name, A, A_norm, A_mean, split, cfg, device):
    Xtr, ytr, Xva, yva, Xte, yte = split[:6]
    aux = {}
    if name == "FFSEEG-GNN":
        model, _top, aux = train_ffseeg(A_norm, Xtr, ytr, Xva, yva, cfg, device)
    else:
        model = model_factory(name, A, A_norm, A_mean, cfg).to(device)
        lr = cfg.lr_fullgcn if name == "FullGraphConv" else cfg.lr_default
        model = _train_generic(model, Xtr, ytr, Xva, yva, cfg, device, lr=lr)
    logits = infer_logits(model, Xte, cfg.batch_size, device)
    m = metrics_from_logits(yte, logits)
    return model, m, aux


def run_repeated_random(df, seeds, models, cfg, device, outdir, tag="random"):
    X = df[CHANNELS].to_numpy(np.float32)
    y = df["eyeDetection"].to_numpy(np.int64)
    A, A_norm, A_mean = build_adjacency(device)
    rows, top_rows = [], []
    for seed in seeds:
        seed_everything(seed)
        split = random_stratified_split(X, y, seed)
        for name in models:
            print(f"[{tag}] seed={seed} model={name}")
            seed_everything(seed)  # paired initialization randomness per model/seed
            _model, m, aux = train_and_eval(name, A, A_norm, A_mean, split, cfg, device)
            row = {"experiment": tag, "seed": seed, "model": name, **m}
            rows.append(row)
            if name == "FFSEEG-GNN" and aux:
                top_rows.append({"seed": seed, **aux})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(outdir / f"{tag}_runs.csv", index=False)
    summarize(rdf).to_csv(outdir / f"{tag}_summary.csv", index=False)
    if top_rows:
        pd.DataFrame(top_rows).to_json(outdir / f"{tag}_ffseeg_topk.json", orient="records", indent=2, force_ascii=False)
    return rdf


def run_grouped(df, seeds, models, cfg, device, outdir, group_size=256, purge_groups=1):
    X = df[CHANNELS].to_numpy(np.float32)
    y = df["eyeDetection"].to_numpy(np.int64)
    A, A_norm, A_mean = build_adjacency(device)
    rows, split_meta = [], []
    for seed in seeds:
        seed_everything(seed)
        split = grouped_time_split(X, y, seed, group_size=group_size, purge_groups=purge_groups)
        split_meta.append({"seed": seed, **split[6]})
        for name in models:
            print(f"[grouped] seed={seed} model={name}")
            seed_everything(seed)
            _model, m, aux = train_and_eval(name, A, A_norm, A_mean, split, cfg, device)
            rows.append({"experiment": "grouped_time", "seed": seed, "model": name, **m})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(outdir / "grouped_time_runs.csv", index=False)
    summarize(rdf).to_csv(outdir / "grouped_time_summary.csv", index=False)
    pd.DataFrame(split_meta).to_csv(outdir / "grouped_time_split_metadata.csv", index=False)
    return rdf


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "auc"]
    out = []
    for model, g in df.groupby("model"):
        r = {"model": model, "n": len(g)}
        for c in metrics:
            r[c + "_mean"] = g[c].mean()
            r[c + "_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        out.append(r)
    return pd.DataFrame(out).sort_values("f1_mean", ascending=False)


def paired_stats(df: pd.DataFrame, reference="FFSEEG-GNN") -> pd.DataFrame:
    rows = []
    pivot = df.pivot_table(index="seed", columns="model", values="f1", aggfunc="first")
    if reference not in pivot.columns:
        return pd.DataFrame()
    pvals = []
    temp = []
    for model in pivot.columns:
        if model == reference:
            continue
        pair = pivot[[reference, model]].dropna()
        d = pair[reference].to_numpy() - pair[model].to_numpy()
        if len(pair) < 2 or np.allclose(d, 0):
            p = 1.0
        else:
            try:
                p = float(wilcoxon(d, alternative="two-sided", method="auto").pvalue)
            except Exception:
                p = np.nan
        # paired standardized mean difference using SD of differences
        dz = float(d.mean() / d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else np.nan
        temp.append((model, len(pair), float(d.mean()), float(d.std(ddof=1)) if len(d)>1 else 0.0, p, dz))
        pvals.append(p)
    # Holm step-down correction
    valid = [(i, p) for i, p in enumerate(pvals) if np.isfinite(p)]
    order = sorted(valid, key=lambda x: x[1])
    holm = [np.nan] * len(pvals)
    m = len(order)
    running = 0.0
    for rank, (idx, p) in enumerate(order):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        holm[idx] = running
    for j, (model, n, md, sd, p, dz) in enumerate(temp):
        rows.append({"reference": reference, "baseline": model, "n_pairs": n, "f1_diff_mean": md,
                     "f1_diff_std": sd, "wilcoxon_p": p, "holm_p": holm[j], "paired_effect_dz": dz})
    return pd.DataFrame(rows)


def run_outlier_sensitivity(data_path, cache_dir, seeds, cfg, device, outdir):
    rows = []
    for keep in [False, True]:
        df, meta = load_eeg(data_path, keep_outliers=keep, smoke_test=False, cache_dir=cache_dir)
        X = df[CHANNELS].to_numpy(np.float32)
        y = df["eyeDetection"].to_numpy(np.int64)
        A, A_norm, A_mean = build_adjacency(device)
        for seed in seeds:
            split = random_stratified_split(X, y, seed)
            for name in ["MLP", "FullGraphConv", "FFSEEG-GNN"]:
                print(f"[outlier] keep={keep} seed={seed} model={name}")
                seed_everything(seed)
                _model, m, _aux = train_and_eval(name, A, A_norm, A_mean, split, cfg, device)
                rows.append({"keep_outliers": keep, "seed": seed, "model": name, **m})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(outdir / "outlier_sensitivity_runs.csv", index=False)
    # compact difference summary
    p = rdf.pivot_table(index=["seed", "model"], columns="keep_outliers", values="f1")
    if False in p.columns and True in p.columns:
        p["delta_clean_minus_all"] = p[False] - p[True]
        p.reset_index().to_csv(outdir / "outlier_sensitivity_f1_deltas.csv", index=False)
    return rdf


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_latency(model: nn.Module, device: torch.device, batch_size: int, warmup=200, reps=2000) -> Dict[str, float]:
    model = model.to(device).eval()
    x = torch.randn(batch_size, 14, 1, device=device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    per_batch_ms = 1000.0 * dt / reps
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": per_batch_ms,
        "latency_ms_per_sample": per_batch_ms / batch_size,
        "throughput_samples_per_s": batch_size * reps / dt,
    }


def run_efficiency(df, cfg, device, outdir, seed=3031):
    # Train representative models on one real split, then benchmark on CPU as requested by manuscript.
    X = df[CHANNELS].to_numpy(np.float32)
    y = df["eyeDetection"].to_numpy(np.int64)
    split = random_stratified_split(X, y, seed)
    A, A_norm, A_mean = build_adjacency(device)
    trained = {}
    for name in ["MLP", "FullGraphConv", "FFSEEG-GNN", "GraphSAGE", "GAT", "GCN-TopKPool", "GCN-SAGPool"]:
        print(f"[efficiency train] {name}")
        seed_everything(seed)
        model, _m, _aux = train_and_eval(name, A, A_norm, A_mean, split, cfg, device)
        trained[name] = model.cpu()
    cpu = torch.device("cpu")
    rows = []
    for name, model in trained.items():
        for bs, reps in [(1, 3000), (64, 1000)]:
            b = benchmark_latency(model, cpu, batch_size=bs, warmup=300, reps=reps)
            rows.append({"model": name, "parameters": count_parameters(model), **b})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(outdir / "efficiency_cpu.csv", index=False)
    return rdf


# ----------------------------- main ---------------------------------------- #

def parse_args():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", type=str, default=None, help="Local EEG Eye State .csv or .arff. If omitted, script downloads it.")
    ap.add_argument("--output", type=str, default="ffseeg_results")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--experiments", nargs="+", default=["random", "grouped", "outlier", "efficiency"],
                    choices=["random", "grouped", "outlier", "efficiency", "all"])
    ap.add_argument("--n-seeds", type=int, default=10, help="Independent seeds for main repeated experiments; use 20 for final manuscript if time allows.")
    ap.add_argument("--start-seed", type=int, default=3031)
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=256, help="Contiguous samples per time group (~2 s if sampling is ~128 Hz).")
    ap.add_argument("--purge-groups", type=int, default=1)
    ap.add_argument("--smoke-test", action="store_true", help="Use synthetic data only to verify code execution; NOT paper data.")
    ap.add_argument("--quick", action="store_true", help="Quick code check: fewer seeds/epochs/models.")
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = outdir / "cache"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)

    n_seeds = 2 if args.quick else args.n_seeds
    seeds = list(range(args.start_seed, args.start_seed + n_seeds))
    cfg = TrainConfig(batch_size=args.batch_size, max_epochs=(20 if args.quick else args.max_epochs), k=args.k)

    df, data_meta = load_eeg(args.data, keep_outliers=False, smoke_test=args.smoke_test, cache_dir=cache_dir)
    metadata = {
        "script": Path(__file__).name,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "device": str(device),
        "data": data_meta,
        "config": asdict(cfg),
        "seeds": seeds,
        "note": "Synthetic smoke-test outputs are not empirical results." if args.smoke_test else "Real public EEG Eye State data expected.",
    }
    (outdir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    experiments = set(args.experiments)
    if "all" in experiments:
        experiments = {"random", "grouped", "outlier", "efficiency"}

    random_df = None
    if "random" in experiments:
        models = ["MLP", "FullGraphConv", "FFSEEG-GNN", "GraphSAGE", "GAT", "GCN-TopKPool", "GCN-SAGPool"]
        if args.quick:
            models = ["MLP", "FullGraphConv", "FFSEEG-GNN"]
        random_df = run_repeated_random(df, seeds, models, cfg, device, outdir, tag="random_stratified")
        ps = paired_stats(random_df, "FFSEEG-GNN")
        ps.to_csv(outdir / "random_stratified_paired_stats.csv", index=False)

    if "grouped" in experiments:
        g_models = ["MLP", "FullGraphConv", "FFSEEG-GNN"]
        grouped_df = run_grouped(df, seeds, g_models, cfg, device, outdir, group_size=args.group_size, purge_groups=args.purge_groups)
        paired_stats(grouped_df, "FFSEEG-GNN").to_csv(outdir / "grouped_time_paired_stats.csv", index=False)

    if "outlier" in experiments:
        if args.smoke_test:
            print("Skipping outlier sensitivity in smoke-test mode (requires the real dataset).")
        else:
            outlier_seeds = seeds[:min(5, len(seeds))]
            run_outlier_sensitivity(args.data, cache_dir, outlier_seeds, cfg, device, outdir)

    if "efficiency" in experiments:
        run_efficiency(df, cfg, device, outdir, seed=seeds[0])

    print("\nFinished. Results saved to:", outdir.resolve())
    if args.smoke_test:
        print("WARNING: This was a SYNTHETIC SMOKE TEST. Do not report these numbers in a paper.")


if __name__ == "__main__":
    main()
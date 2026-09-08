"""
run_validity_baselines.py — Detectors + trivial baselines + threshold sweeps on huflit_v2.
Uses only data/processed/huflit_v2/*.npy (no raw FortiGate I/O).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype"))
from pipeline.tcn_transformer import (  # noqa: E402
    SinusoidalPositionEncoding,
    TCNTransformerAutoencoder,
    make_masked_inputs,
    masked_token_loss,
    pseudo_log_likelihood_scores,
)

SEEDS = (42, 101, 202, 303, 404)
DEVICE = torch.device("cpu")
DATA = ROOT / "data" / "processed" / "huflit_v2"
OUT = ROOT / "results" / "tables" / "huflit_v2_baselines.json"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split():
    X_tr = np.load(DATA / "train_sequences.npy")
    y_tr = np.load(DATA / "train_labels.npy")
    X_va = np.load(DATA / "val_sequences.npy")
    y_va = np.load(DATA / "val_labels.npy")
    X_te = np.load(DATA / "test_sequences.npy")
    y_te = np.load(DATA / "test_labels.npy")
    with open(DATA / "templates.json", encoding="utf-8") as f:
        tmpl = json.load(f)
    vocab = max(tmpl.values()) + 2
    return X_tr, y_tr, X_va, y_va, X_te, y_te, vocab


def metrics_at(scores, y, thr):
    pred = (scores >= thr).astype(int)
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "prevalence": float(np.mean(y)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, scores))
    except ValueError:
        out["auprc"] = float("nan")
    prev = out["prevalence"]
    out["auprc_lift_over_prevalence"] = (
        float(out["auprc"] - prev) if out["auprc"] == out["auprc"] else float("nan")
    )
    always_f1 = 2 * prev / (1 + prev) if prev > 0 else 0.0
    out["always_anomaly_f1"] = float(always_f1)
    out["f1_gap_vs_always_anomaly"] = float(out["f1"] - always_f1)
    return out


def nominal_threshold(scores, y, mode="mu_sigma", k=3.0, pct=95.0):
    nom = scores[y == 0]
    if len(nom) < 10:
        nom = scores
    if mode == "mu_sigma":
        return float(np.mean(nom) + k * np.std(nom))
    if mode == "percentile":
        return float(np.percentile(nom, pct))
    if mode == "oracle_f1":
        # diagnostic only: pick thr maximizing F1 on validation labels
        best_thr, best_f1 = float(np.median(scores)), -1.0
        for q in np.linspace(0.05, 0.995, 60):
            thr = float(np.quantile(scores, q))
            f1 = f1_score(y, (scores >= thr).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = float(f1), thr
        return best_thr
    raise ValueError(mode)


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and not (isinstance(r[key], float) and np.isnan(r[key]))]
    return {
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "values": vals,
    }


class TransformerNoTCN(nn.Module):
    def __init__(self, vocab_size, hidden=128, num_heads=4, seq_len=20):
        super().__init__()
        self.pad_id = 0
        self.mask_token_id = vocab_size - 1
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = SinusoidalPositionEncoding(hidden, seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=256, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        pad_mask = x.eq(self.pad_id)
        h = self.position(self.emb(x))
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.fc(h)


def train_masked(model, X_norm, epochs, batch_size, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    curve = []
    t0 = time.time()
    n = len(X_norm)
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        total, steps = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = torch.tensor(X_norm[idx], dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            masked, selected = make_masked_inputs(batch, model.mask_token_id, 0.15)
            loss = masked_token_loss(model(masked), batch, selected)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        curve.append(total / max(steps, 1))
        print(f"      ep {ep}/{epochs} loss={curve[-1]:.4f}")
    return time.time() - t0, curve


@torch.no_grad()
def score_pll(model, X, batch_size=256):
    model.eval()
    scores = []
    t0 = time.perf_counter()
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i : i + batch_size], dtype=torch.long, device=DEVICE)
        scores.append(pseudo_log_likelihood_scores(model, batch, model.mask_token_id).cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(scores), (elapsed / max(len(X), 1)) * 1000.0


@torch.no_grad()
def batch1_pll_ms(model, X, samples=50):
    model.eval()
    subset = X[: min(samples, len(X))]
    for row in subset[:5]:
        score_pll(model, row[None, :], batch_size=1)
    t0 = time.perf_counter()
    for row in subset:
        score_pll(model, row[None, :], batch_size=1)
    return (time.perf_counter() - t0) * 1000.0 / max(len(subset), 1)


def count_vectorize(X, vocab):
    out = np.zeros((len(X), vocab), dtype=np.float32)
    for i, row in enumerate(X):
        for t in row:
            if 0 < t < vocab:
                out[i, t] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out / n


def eval_pca(X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, seed):
    set_seed(seed)
    Xn = X_tr[y_tr == 0] if (y_tr == 0).sum() >= 50 else X_tr
    pca = PCA(n_components=min(32, len(Xn) - 1, vocab - 1), random_state=seed)
    Cn = count_vectorize(Xn, vocab)
    pca.fit(Cn)

    def err(X):
        C = count_vectorize(X, vocab)
        R = pca.inverse_transform(pca.transform(C))
        return np.mean((C - R) ** 2, axis=1)

    s_va, s_te = err(X_va), err(X_te)
    return s_va, s_te


def threshold_sweep(s_va, y_va, s_te, y_te):
    rows = {}
    for k in (1.0, 2.0, 3.0):
        thr = nominal_threshold(s_va, y_va, "mu_sigma", k=k)
        m = metrics_at(s_te, y_te, thr)
        m["threshold"] = thr
        rows[f"mu_plus_{k:g}sigma"] = m
    for pct in (95.0, 97.5, 99.0):
        thr = nominal_threshold(s_va, y_va, "percentile", pct=pct)
        m = metrics_at(s_te, y_te, thr)
        m["threshold"] = thr
        rows[f"nominal_p{pct:g}"] = m
    # oracle on VAL then apply to test (diagnostic upper bound, not deployable claim)
    thr_o = nominal_threshold(s_va, y_va, "oracle_f1")
    m = metrics_at(s_te, y_te, thr_o)
    m["threshold"] = thr_o
    m["note"] = "oracle thr chosen on validation labels; diagnostic only"
    rows["oracle_val_f1_thr"] = m
    return rows


def trivial_baselines(y_te):
    prev = float(np.mean(y_te))
    # always anomaly
    scores_always = np.ones(len(y_te))
    m_always = metrics_at(scores_always, y_te, 0.5)
    # random scores
    rng = np.random.default_rng(0)
    scores_rand = rng.random(len(y_te))
    m_rand = metrics_at(scores_rand, y_te, 0.5)
    try:
        m_rand["roc_auc"] = float(roc_auc_score(y_te, scores_rand))
        m_rand["auprc"] = float(average_precision_score(y_te, scores_rand))
    except ValueError:
        pass
    m_rand["auprc_lift_over_prevalence"] = float(m_rand.get("auprc", prev) - prev)
    return {
        "prevalence": prev,
        "always_anomaly": m_always,
        "random_scores_at_0.5": m_rand,
        "no_skill_auprc": prev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--models", default="PCA,Masked Transformer,Proposed")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    print("=" * 72)
    print(f" VALIDITY BASELINES — huflit_v2 — threads={args.threads}")
    print("=" * 72)

    X_tr, y_tr, X_va, y_va, X_te, y_te, vocab = load_split()
    print(
        f" vocab={vocab} train={len(X_tr)} val={len(X_va)} test={len(X_te)} "
        f"test_anom={100*y_te.mean():.1f}%"
    )

    results = {
        "experiment": "huflit_v2_validity_baselines",
        "data": str(DATA),
        "protocol": {
            "split": "day-based after global line-hash dedup; windows source/member-bounded",
            "window_label": "OR over L proxy line labels",
            "train": "nominal-only masked CE for NN models",
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "threads": args.threads,
            "scoring": "leave-one-out PLL for masked models",
        },
        "trivial_baselines": trivial_baselines(y_te),
        "models": {},
    }

    want = {x.strip().lower() for x in args.models.split(",") if x.strip()}

    # PCA
    if any(w.startswith("pca") for w in want):
        print("\n### PCA")
        rows, sweeps = [], []
        last_scores = None
        for seed in SEEDS:
            print(f"    seed={seed}")
            s_va, s_te = eval_pca(X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, seed)
            thr = nominal_threshold(s_va, y_va, "mu_sigma", k=3)
            m = metrics_at(s_te, y_te, thr)
            m["seed"] = seed
            m["threshold"] = thr
            rows.append(m)
            sweeps.append(threshold_sweep(s_va, y_va, s_te, y_te))
            last_scores = s_te
            print(f"      F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} AUPRC={m['auprc']:.4f} "
                  f"lift={m['auprc_lift_over_prevalence']:.4f}")
        results["models"]["PCA (Count Vectors)"] = {
            "primary_mu3sigma": {k: agg(rows, k) for k in
                                 ["precision", "recall", "f1", "roc_auc", "auprc",
                                  "auprc_lift_over_prevalence", "f1_gap_vs_always_anomaly"]},
            "threshold_sweeps_per_seed": sweeps,
        }

    def run_masked(name, factory):
        print(f"\n### {name}")
        rows, sweeps, curves, lats = [], [], [], []
        for seed in SEEDS:
            set_seed(seed)
            model = factory().to(DEVICE)
            Xn = X_tr[y_tr == 0]
            if len(Xn) < 100:
                Xn = X_tr
            print(f"    [{name}] seed={seed}")
            _, curve = train_masked(model, Xn, args.epochs, args.batch_size)
            s_va, _ = score_pll(model, X_va)
            s_te, batched = score_pll(model, X_te)
            b1 = batch1_pll_ms(model, X_te)
            thr = nominal_threshold(s_va, y_va, "mu_sigma", k=3)
            m = metrics_at(s_te, y_te, thr)
            m.update({"seed": seed, "threshold": thr, "batch1_ms": b1, "batched_ms": batched})
            rows.append(m)
            sweeps.append(threshold_sweep(s_va, y_va, s_te, y_te))
            curves.append(curve)
            lats.append({"batch1_ms": b1, "batched_ms": batched})
            print(
                f"      F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} AUPRC={m['auprc']:.4f} "
                f"lift={m['auprc_lift_over_prevalence']:.4f} batch1={b1:.2f}ms"
            )
        block = {
            "primary_mu3sigma": {k: agg(rows, k) for k in
                                 ["precision", "recall", "f1", "roc_auc", "auprc",
                                  "auprc_lift_over_prevalence", "f1_gap_vs_always_anomaly",
                                  "batch1_ms", "batched_ms"]},
            "threshold_sweeps_per_seed": sweeps,
            "train_loss_curves": curves,
        }
        results["models"][name] = block

    if any("masked" in w and "proposed" not in w for w in want) or any(w == "masked transformer" for w in want):
        run_masked(
            "Masked Transformer (No TCN)",
            lambda: TransformerNoTCN(vocab, seq_len=X_tr.shape[1]),
        )
    if any("proposed" in w for w in want):
        run_masked(
            "Proposed Masked TCN-Transformer",
            lambda: TCNTransformerAutoencoder(vocab_size=vocab, seq_len=X_tr.shape[1]),
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

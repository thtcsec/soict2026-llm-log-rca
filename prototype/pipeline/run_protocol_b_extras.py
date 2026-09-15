"""
run_protocol_b_extras.py
========================
Add DeepLog + LogAnomaly-lite to Protocol B, and a keyword/status template-ablation
study (PCA + Masked Transformer). Merges into huflit_v2_baselines.json.
Uses only data/processed/huflit_v2 (no FortiGate I/O).
"""

from __future__ import annotations

import argparse
import json
import re
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
    make_masked_inputs,
    masked_token_loss,
    pseudo_log_likelihood_scores,
)
from pipeline.run_validity_baselines import (  # noqa: E402
    TransformerNoTCN,
    agg,
    batch1_pll_ms,
    metrics_at,
    nominal_threshold,
    score_pll,
    set_seed,
    threshold_sweep,
    train_masked,
    trivial_baselines,
)

SEEDS = (42, 101, 202, 303, 404)
DEVICE = torch.device("cpu")
DATA = ROOT / "data" / "processed" / "huflit_v2"
OUT = ROOT / "results" / "tables" / "huflit_v2_baselines.json"

TRIGGER = re.compile(
    r"(?:\b(?:4\d{2}|5\d{2})\b|deny|block|error|fail|critical|attack|unauthorized|"
    r"forbidden|refused|invalid|malware|intrusion|alert|drop|reject)",
    re.I,
)


class DeepLogLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x[:, :-1]))
        return self.fc(h)


class LogAnomalyLite(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.fc(h)


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
    return X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, tmpl


def count_vectorize(X, vocab):
    out = np.zeros((len(X), vocab), dtype=np.float32)
    for i, row in enumerate(X):
        for t in row:
            if 0 < t < vocab:
                out[i, t] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out / n


def train_seq(model, X_norm, epochs, batch_size, mode: str, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(X_norm)
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        total, steps = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = torch.tensor(X_norm[idx], dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            if mode == "deeplog":
                logits = model(batch)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1)
                )
            else:
                logits = model(batch)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch.reshape(-1)
                )
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(f"      ep {ep}/{epochs} loss={total/max(steps,1):.4f}")


@torch.no_grad()
def score_seq(model, X, mode: str, batch_size=512):
    model.eval()
    scores = []
    t0 = time.perf_counter()
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i : i + batch_size], dtype=torch.long, device=DEVICE)
        if mode == "deeplog":
            logits = model(batch)
            nll = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                batch[:, 1:].reshape(-1),
                reduction="none",
            ).view(batch.size(0), -1)
            scores.append(nll.mean(dim=1).cpu().numpy())
        else:
            logits = model(batch)
            nll = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch.reshape(-1), reduction="none"
            ).view(batch.size(0), batch.size(1))
            scores.append(nll.mean(dim=1).cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(scores), (elapsed / max(len(X), 1)) * 1000.0


def eval_nn_block(name, factory, mode, X_tr, y_tr, X_va, y_va, X_te, y_te, epochs, bs):
    print(f"\n### {name}")
    rows, sweeps = [], []
    for seed in SEEDS:
        set_seed(seed)
        model = factory().to(DEVICE)
        Xn = X_tr[y_tr == 0]
        if len(Xn) < 100:
            Xn = X_tr
        print(f"    [{name}] seed={seed}")
        train_seq(model, Xn, epochs, bs, mode)
        s_va, _ = score_seq(model, X_va, mode)
        s_te, batched = score_seq(model, X_te, mode)
        thr = nominal_threshold(s_va, y_va, "mu_sigma", k=3)
        m = metrics_at(s_te, y_te, thr)
        m.update({"seed": seed, "threshold": thr, "batched_ms": batched})
        rows.append(m)
        sweeps.append(threshold_sweep(s_va, y_va, s_te, y_te))
        print(
            f"      F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} AUPRC={m['auprc']:.4f} "
            f"lift={m['auprc_lift_over_prevalence']:.4f}"
        )
    return {
        "primary_mu3sigma": {
            k: agg(rows, k)
            for k in [
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "auprc",
                "auprc_lift_over_prevalence",
                "f1_gap_vs_always_anomaly",
                "batched_ms",
            ]
        },
        "threshold_sweeps_per_seed": sweeps,
    }


def trigger_template_ids(tmpl: dict) -> set[int]:
    ids = set()
    for text, tid in tmpl.items():
        if TRIGGER.search(text):
            ids.add(int(tid))
    return ids


def ablate_sequences(X: np.ndarray, trigger_ids: set[int], replacement: int = 0) -> np.ndarray:
    Y = X.copy()
    for tid in trigger_ids:
        Y[Y == tid] = replacement
    return Y


def eval_pca_scores(X_tr, y_tr, X_va, X_te, vocab, seed):
    set_seed(seed)
    Xn = X_tr[y_tr == 0] if (y_tr == 0).sum() >= 50 else X_tr
    pca = PCA(n_components=min(32, len(Xn) - 1, vocab - 1), random_state=seed)
    Cn = count_vectorize(Xn, vocab)
    pca.fit(Cn)

    def err(X):
        C = count_vectorize(X, vocab)
        R = pca.inverse_transform(pca.transform(C))
        return np.mean((C - R) ** 2, axis=1)

    return err(X_va), err(X_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--skip_ablation", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, tmpl = load_split()
    print("=" * 72)
    print(f" Protocol B extras — vocab={vocab} test_anom={100*y_te.mean():.1f}%")
    print("=" * 72)

    if OUT.exists():
        payload = json.loads(OUT.read_text(encoding="utf-8"))
    else:
        payload = {
            "experiment": "huflit_v2_validity_baselines",
            "data": str(DATA),
            "trivial_baselines": trivial_baselines(y_te),
            "models": {},
        }

    payload["models"]["DeepLog (LSTM)"] = eval_nn_block(
        "DeepLog (LSTM)",
        lambda: DeepLogLSTM(vocab),
        "deeplog",
        X_tr,
        y_tr,
        X_va,
        y_va,
        X_te,
        y_te,
        args.epochs,
        args.batch_size,
    )
    payload["models"]["LogAnomaly-lite (Emb+LSTM)"] = eval_nn_block(
        "LogAnomaly-lite (Emb+LSTM)",
        lambda: LogAnomalyLite(vocab),
        "recon",
        X_tr,
        y_tr,
        X_va,
        y_va,
        X_te,
        y_te,
        args.epochs,
        args.batch_size,
    )

    if not args.skip_ablation:
        trig = trigger_template_ids(tmpl)
        print(f"\n### Keyword/status template ablation — {len(trig)} trigger template IDs")
        X_tr_a = ablate_sequences(X_tr, trig)
        X_va_a = ablate_sequences(X_va, trig)
        X_te_a = ablate_sequences(X_te, trig)
        # Fraction of tokens ablated on test
        frac = float(np.mean(np.isin(X_te, list(trig)))) if trig else 0.0
        abl = {
            "n_trigger_templates": len(trig),
            "test_token_ablation_fraction": frac,
            "note": (
                "Templates whose string matches HTTP 4xx/5xx or deny/block/error-like "
                "keywords used in proxy labeling are replaced by pad id=0 before scoring."
            ),
            "models": {},
        }
        # PCA
        rows = []
        for seed in SEEDS:
            s_va, s_te = eval_pca_scores(X_tr_a, y_tr, X_va_a, X_te_a, vocab, seed)
            thr = nominal_threshold(s_va, y_va, "mu_sigma", k=3)
            m = metrics_at(s_te, y_te, thr)
            m["seed"] = seed
            rows.append(m)
            print(f"    PCA-ablated seed={seed} AUC={m['roc_auc']:.4f} lift={m['auprc_lift_over_prevalence']:.4f}")
        abl["models"]["PCA (ablated)"] = {
            "primary_mu3sigma": {
                k: agg(rows, k)
                for k in [
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "auprc",
                    "auprc_lift_over_prevalence",
                ]
            }
        }
        # Masked Transformer (3 seeds for time) — still report clearly
        rows = []
        for seed in SEEDS:
            set_seed(seed)
            model = TransformerNoTCN(vocab, seq_len=X_tr.shape[1]).to(DEVICE)
            Xn = X_tr_a[y_tr == 0]
            if len(Xn) < 100:
                Xn = X_tr_a
            print(f"    [Masked Transformer ablated] seed={seed}")
            train_masked(model, Xn, args.epochs, args.batch_size)
            s_va, _ = score_pll(model, X_va_a)
            s_te, _ = score_pll(model, X_te_a)
            thr = nominal_threshold(s_va, y_va, "mu_sigma", k=3)
            m = metrics_at(s_te, y_te, thr)
            m["seed"] = seed
            rows.append(m)
            print(f"      AUC={m['roc_auc']:.4f} lift={m['auprc_lift_over_prevalence']:.4f}")
        abl["models"]["Masked Transformer (No TCN, ablated)"] = {
            "primary_mu3sigma": {
                k: agg(rows, k)
                for k in [
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "auprc",
                    "auprc_lift_over_prevalence",
                ]
            }
        }
        payload["keyword_status_ablation"] = abl

    payload["protocol_b_extras_note"] = (
        "DeepLog and LogAnomaly-lite added on the same Protocol B arrays; "
        "ablation remaps proxy-trigger templates to pad before scoring."
    )
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

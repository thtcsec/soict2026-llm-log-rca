"""
run_baselines_huflit.py — Measured baselines on the SAME hardened HUFLIT split.
Models: PCA, DeepLog (LSTM next-token), LogAnomaly-lite (emb+LSTM recon),
        masked Transformer without TCN, Proposed Masked TCN-Transformer.
Protocol mirrors harden_huflit_train.py: nominal-only train, τ=μ+3σ on val, 5 seeds.
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
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA = ROOT / "data" / "processed" / "huflit"
OUT = ROOT / "results" / "tables" / "huflit_baselines.json"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def nominal_mu_sigma_threshold(scores, y, k=3.0):
    nom = scores[y == 0]
    if len(nom) < 10:
        nom = scores
    mu, sigma = float(np.mean(nom)), float(np.std(nom))
    return mu + k * sigma


def metrics_at(scores, y, thr):
    pred = (scores >= thr).astype(int)
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y, scores))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["auprc"] = float(average_precision_score(y, scores))
    except ValueError:
        out["auprc"] = float("nan")
    return out


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
    return {
        "mean": float(np.mean(vals)) if vals else float("nan"),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "values": vals,
    }


# ─── Models ───────────────────────────────────────────────────────────────────


class DeepLogLSTM(nn.Module):
    """Classic DeepLog: predict next template from history."""

    def __init__(self, vocab_size, embed_dim=64, hidden=128, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        # x: [B, L] — use first L-1 to predict last L-1 (shifted)
        h, _ = self.lstm(self.emb(x[:, :-1]))
        return self.fc(h)  # [B, L-1, V]


class LogAnomalyLite(nn.Module):
    """LogAnomaly-inspired: embedding + LSTM sequence reconstruction."""

    def __init__(self, vocab_size, embed_dim=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.fc(h)


class TransformerAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden=128, seq_len=20):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)
        self.position = SinusoidalPositionEncoding(hidden, seq_len)
        self.mask_token_id = vocab_size - 1
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=num_heads, dim_feedforward=256, batch_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        pad_mask = x.eq(0)
        return self.fc(self.enc(self.position(self.emb(x)), src_key_padding_mask=pad_mask))


def count_vectorize(X, vocab_size):
    # bag-of-template counts per window
    B, L = X.shape
    out = np.zeros((B, vocab_size), dtype=np.float32)
    for i in range(B):
        for t in X[i]:
            if 0 <= t < vocab_size:
                out[i, t] += 1.0
    # L2 normalize
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out / n


def train_torch(model, X_norm, epochs, batch_size, lr, mode: str):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = len(X_norm)
    t0 = time.time()
    curve = []
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
            elif mode == "masked":
                masked, selected = make_masked_inputs(batch, model.mask_token_id, 0.15)
                loss = masked_token_loss(model(masked), batch, selected)
            else:  # full recon
                logits = model(batch)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), batch.reshape(-1)
                )
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        print(f"      ep {ep}/{epochs} loss={total/max(steps,1):.4f}")
        curve.append(total / max(steps, 1))
    return time.time() - t0, curve


@torch.no_grad()
def score_torch(model, X, mode: str, batch_size=512):
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
        elif mode == "masked":
            scores.append(
                pseudo_log_likelihood_scores(model, batch, model.mask_token_id).cpu().numpy()
            )
        else:
            logits = model(batch)
            nll = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch.reshape(-1), reduction="none"
            ).view(batch.size(0), batch.size(1))
            scores.append(nll.mean(dim=1).cpu().numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(scores), (elapsed / max(len(X), 1)) * 1000.0


@torch.no_grad()
def benchmark_batch1(model, X, mode: str, warmup=10, samples=100):
    """Wall-clock single-sequence latency; reported separately from batched throughput."""
    model.eval()
    subset = X[: min(samples, len(X))]
    if not len(subset):
        return float("nan")
    for row in subset[: min(warmup, len(subset))]:
        score_torch(model, row[None, :], mode, batch_size=1)
    start = time.perf_counter()
    for row in subset:
        score_torch(model, row[None, :], mode, batch_size=1)
    return (time.perf_counter() - start) * 1000.0 / len(subset)


def eval_pca(X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, seed):
    set_seed(seed)
    Xn = X_tr[y_tr == 0]
    C_n = count_vectorize(Xn, vocab)
    C_va = count_vectorize(X_va, vocab)
    C_te = count_vectorize(X_te, vocab)
    n_comp = min(32, C_n.shape[0] - 1, C_n.shape[1] - 1)
    t0 = time.time()
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(C_n)
    train_s = time.time() - t0

    def recon_err(C):
        Z = pca.transform(C)
        R = pca.inverse_transform(Z)
        return np.mean((C - R) ** 2, axis=1)

    t1 = time.perf_counter()
    s_va = recon_err(C_va)
    s_te = recon_err(C_te)
    batched_lat = (time.perf_counter() - t1) / max(len(X_te) + len(X_va), 1) * 1000.0
    sample = X_te[: min(100, len(X_te))]
    t2 = time.perf_counter()
    for row in sample:
        recon_err(count_vectorize(row[None, :], vocab))
    batch1_lat = (time.perf_counter() - t2) / max(len(sample), 1) * 1000.0
    thr = nominal_mu_sigma_threshold(s_va, y_va)
    m = metrics_at(s_te, y_te, thr)
    m.update({"seed": seed, "threshold": thr, "batch1_latency_ms": batch1_lat, "batched_ms_per_seq": batched_lat, "train_seconds": train_s})
    return m


def eval_nn(name, factory, mode, X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, seed, epochs, bs):
    set_seed(seed)
    model = factory().to(DEVICE)
    Xn = X_tr[y_tr == 0]
    if len(Xn) < 100:
        Xn = X_tr
    print(f"    [{name}] seed={seed}")
    train_s, curve = train_torch(model, Xn, epochs, bs, 1e-3, mode)
    s_va, _ = score_torch(model, X_va, mode)
    s_te, batched_lat = score_torch(model, X_te, mode)
    batch1_lat = benchmark_batch1(model, X_te, mode)
    thr = nominal_mu_sigma_threshold(s_va, y_va)
    m = metrics_at(s_te, y_te, thr)
    m.update({
        "seed": seed,
        "threshold": thr,
        "batch1_latency_ms": batch1_lat,
        "batched_ms_per_seq": batched_lat,
        "train_seconds": train_s,
    })
    print(
        f"      F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} P={m['precision']:.4f} "
        f"R={m['recall']:.4f} batch1={batch1_lat:.3f}ms batched={batched_lat:.3f}ms/seq"
    )
    return m, s_te, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument(
        "--models",
        default="all",
        help="Comma-separated model-name substrings; existing JSON blocks are retained for models not rerun.",
    )
    args = ap.parse_args()
    epochs = args.epochs
    bs = args.batch_size
    torch.set_num_threads(max(1, args.threads))
    torch.set_num_interop_threads(1)
    print("=" * 72)
    print(f" HUFLIT BASELINES — device={DEVICE}")
    print("=" * 72)
    X_tr, y_tr, X_va, y_va, X_te, y_te, vocab = load_split()
    seq_len = X_tr.shape[1]
    print(
        f" vocab={vocab} seq_len={seq_len} "
        f"train={len(X_tr)} val={len(X_va)} test={len(X_te)} "
        f"test_anom={y_te.mean()*100:.1f}%"
    )

    specs = [
        ("PCA (Count Vectors)", None, "pca"),
        (
            "DeepLog (LSTM)",
            lambda: DeepLogLSTM(vocab),
            "deeplog",
        ),
        (
            "LogAnomaly-lite (Emb+LSTM)",
            lambda: LogAnomalyLite(vocab),
            "masked",
        ),
        (
            "Masked Transformer (No TCN)",
            lambda: TransformerAE(vocab, seq_len=seq_len),
            "recon",
        ),
        (
            "Proposed Masked TCN-Transformer",
            lambda: TCNTransformerAutoencoder(vocab_size=vocab, seq_len=seq_len),
            "masked",
        ),
    ]

    all_results = {}
    prior_diagnostics = None
    if OUT.exists() and args.models != "all":
        with open(OUT, encoding="utf-8") as f:
            prior = json.load(f)
            all_results.update(prior.get("models", {}))
            prior_diagnostics = prior.get("diagnostics", {}).get("proposed")
        if "Transformer AE (No TCN)" in all_results:
            all_results["Masked Transformer (No TCN)"] = all_results.pop("Transformer AE (No TCN)")
        for block in all_results.values():
            if "batch1_latency_ms" not in block:
                block["legacy_batched_latency_ms_per_seq"] = block.get("latency_ms_per_seq")
                block["batch1_latency_ms"] = {"mean": None, "std": None, "values": []}
                block["batched_ms_per_seq"] = block.get("latency_ms_per_seq", {"mean": None, "std": None, "values": []})
    proposed_diagnostics = prior_diagnostics
    for name, factory, mode in specs:
        selected = args.models == "all" or any(
            token.strip().lower() in name.lower() for token in args.models.split(",")
        )
        if not selected:
            print(f"\n### {name} [retained; not rerun]")
            continue
        print(f"\n### {name}")
        rows = []
        curves = []
        for seed in SEEDS:
            if mode == "pca":
                row = eval_pca(X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, seed)
                print(
                    f"    seed={seed} F1={row['f1']:.4f} AUC={row['roc_auc']:.4f} "
                    f"P={row['precision']:.4f} R={row['recall']:.4f}"
                )
            else:
                row, test_scores, curve = eval_nn(
                    name, factory, mode, X_tr, y_tr, X_va, y_va, X_te, y_te, vocab, seed, epochs, bs
                )
                curves.append(curve)
            rows.append(row)
            if name == "Proposed Masked TCN-Transformer" and seed == SEEDS[-1]:
                fpr, tpr, _ = roc_curve(y_te, test_scores)
                prec, rec, _ = precision_recall_curve(y_te, test_scores)
                proposed_diagnostics = {
                    "representative_seed": seed,
                    "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
                    "pr_curve": {"precision": prec.tolist(), "recall": rec.tolist()},
                    "representative_metrics": metrics_at(test_scores, y_te, row["threshold"]),
                }
        all_results[name] = {
            "precision": agg(rows, "precision"),
            "recall": agg(rows, "recall"),
            "f1": agg(rows, "f1"),
            "roc_auc": agg(rows, "roc_auc"),
            "auprc": agg(rows, "auprc"),
            "batch1_latency_ms": agg(rows, "batch1_latency_ms"),
            "batched_ms_per_seq": agg(rows, "batched_ms_per_seq"),
            "per_seed": rows,
        }
        if curves:
            all_results[name]["mean_train_loss_curve"] = np.mean(np.asarray(curves), axis=0).tolist()

    payload = {
        "experiment": "huflit_baselines_same_split",
        "device": str(DEVICE),
        "protocol": {
            "split": "precomputed stream-position 80/10/10 before windowing; backup-duplicate overlap is documented",
            "train": "nominal-only",
            "threshold": "tau = mu_val + 3*sigma_val on nominal val",
            "seeds": list(SEEDS),
            "epochs": epochs,
            "data": str(DATA),
            "threads": args.threads,
            "scoring": "leave-one-position-out pseudo-log-likelihood for masked models",
            "latency": "batch=1 wall clock reported separately from batch=512 throughput-equivalent time",
            "rerun_scope": args.models,
            "retained_models_note": "When --models is partial, unchanged accuracy rows are retained; legacy batched latency is not relabeled as batch=1.",
        },
        "models": all_results,
        "diagnostics": {"proposed": proposed_diagnostics},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"{'Model':<36} | {'F1':>10} | {'AUC':>10} | {'P':>10} | {'R':>10} | {'ms':>8}")
    print("-" * 88)
    for name, block in all_results.items():
        latency = block["batch1_latency_ms"]["mean"]
        latency_text = f"{latency:7.4f}" if latency is not None else "     NR"
        print(
            f"{name:<36} | {block['f1']['mean']:6.4f}±{block['f1']['std']:.3f} | "
            f"{block['roc_auc']['mean']:6.4f}±{block['roc_auc']['std']:.3f} | "
            f"{block['precision']['mean']:6.4f}±{block['precision']['std']:.3f} | "
            f"{block['recall']['mean']:6.4f}±{block['recall']['std']:.3f} | "
            + latency_text
        )
    print("=" * 88)
    print(f"[+] Saved {OUT}")


if __name__ == "__main__":
    main()

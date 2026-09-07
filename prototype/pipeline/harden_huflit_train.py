"""
harden_huflit_train.py — Honest HUFLIT experiment for SOICT 2026
================================================================
- Streams selected log members from day zips (NO full RAR dump)
- Excludes all FortiGate members and oversized members by default
- Stream-position 80/10/10 split BEFORE windowing
- Semi-supervised: train Masked TCN-Transformer on nominal sequences only
- Val-set threshold; 5 seeds; write measured JSON + figure arrays
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    TCNTransformerAutoencoder,
    make_masked_inputs,
    masked_token_loss,
    pseudo_log_likelihood_scores,
)

SEEDS = (42, 101, 202, 303, 404)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANOM_KW = re.compile(
    r"\b(error|fail|denied|critical|refused|invalid|attack|flood|drop|reject|"
    r"alert|blocked|unauthorized|forbidden|malware|virus|intrusion|bruteforce|"
    r"action=\"deny\"|action=deny|status=deny)\b",
    re.I,
)
HTTP_STATUS = re.compile(r"\s(\d{3})\s")
FORTI_DENY = re.compile(r'action[=:]"?deny"?|utmaction[=:]"?block"?|attack[=:]"?', re.I)


class SimilarityTemplateParser:
    """Lightweight similarity parser; this is not the Drain3 implementation."""
    MASK_TOKENS = [
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
        (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "<UUID>"),
        (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
        (re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"), "<TS>"),
        (re.compile(r"\b\d+\.\d+\b"), "<FLOAT>"),
        (re.compile(r"\b\d{5,}\b"), "<NUM>"),
        (re.compile(r"/(?:home|var|usr|etc|tmp|proc|opt)/\S+"), "<PATH>"),
    ]

    def __init__(self, sim_th: float = 0.5, max_vocab: int = 2500):
        self.sim_th = sim_th
        self.max_vocab = max_vocab
        self.templates: list[list[str]] = []
        self.template_to_id: dict[str, int] = {}

    def _sanitize(self, msg: str) -> str:
        for pat, rep in self.MASK_TOKENS:
            msg = pat.sub(rep, msg)
        return msg.strip()

    def _similarity(self, a: list[str], b: list[str]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        return sum(x == y for x, y in zip(a, b)) / len(a)

    def parse(self, raw_line: str) -> int:
        tokens = self._sanitize(raw_line).split()
        if not tokens:
            return 0
        best_idx, best_sim = -1, -1.0
        for i, tmpl in enumerate(self.templates):
            sim = self._similarity(tokens, tmpl)
            if sim > best_sim and sim >= self.sim_th:
                best_sim, best_idx = sim, i
        if best_idx >= 0:
            merged = [a if a == b else "<*>" for a, b in zip(tokens, self.templates[best_idx])]
            self.templates[best_idx] = merged
            tmpl_str = " ".join(merged)
        else:
            if len(self.templates) >= self.max_vocab:
                best_idx = min(range(len(self.templates)), key=lambda i: i)
                tmpl_str = " ".join(self.templates[best_idx])
            else:
                self.templates.append(tokens)
                tmpl_str = " ".join(tokens)
        if tmpl_str not in self.template_to_id:
            self.template_to_id[tmpl_str] = len(self.template_to_id) + 1
        return self.template_to_id[tmpl_str]


def is_anomaly_line(line: str) -> bool:
    m = HTTP_STATUS.search(line)
    if m:
        code = int(m.group(1))
        if 400 <= code <= 599:
            return True
    if FORTI_DENY.search(line):
        return True
    return bool(ANOM_KW.search(line))


def discover_log_members(root: Path, max_member_mb: float, include_fortigate: bool = False) -> list[tuple[Path, str, int]]:
    """Return members in deterministic backup-day/ZIP/member order."""
    items: list[tuple[str, Path, str, int]] = []
    for zpath in sorted(root.rglob("*.zip")):
        day = zpath.parent.name
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                for info in zf.infolist():
                    name = info.filename.replace("\\", "/")
                    if info.is_dir():
                        continue
                    if not include_fortigate and "forti" in (str(zpath) + "/" + name).lower():
                        continue
                    if not name.lower().endswith(".log"):
                        continue
                    if "fortigate.csv" in name.lower():
                        continue
                    size_mb = info.file_size / (1024 * 1024)
                    if size_mb > max_member_mb:
                        print(f"  [skip large] {zpath.name}::{name} ({size_mb:.0f} MB)")
                        continue
                    items.append((day, zpath, name, info.file_size))
        except zipfile.BadZipFile as e:
            print(f"  [bad zip] {zpath}: {e}")
    items.sort(key=lambda t: (t[0], t[1].name, t[2]))
    return [(z, m, s) for _, z, m, s in items]


def stream_labeled_lines(
    members: list[tuple[Path, str, int]],
    max_lines: int,
    per_file_cap: int,
):
    n = 0
    bytes_read = 0
    files_used = 0
    for zpath, member, size in members:
        if n >= max_lines:
            break
        files_used += 1
        local = 0
        print(f"  • {zpath.parent.name}/{zpath.name} :: {member} ({size/1e6:.1f} MB)")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                with zf.open(member, "r") as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    for line in text:
                        line = line.strip()
                        if not line:
                            continue
                        yield is_anomaly_line(line), line
                        n += 1
                        local += 1
                        bytes_read += len(line) + 1
                        if n >= max_lines or local >= per_file_cap:
                            break
        except Exception as e:
            print(f"    [!] {e}")
        time.sleep(0.05)  # gentle pause between zip members
    meta = {"lines": n, "bytes_read": bytes_read, "files_used": files_used}
    return meta


def build_token_stream(members, max_lines, per_file_cap, max_vocab, per_day_cap: int):
    parser = SimilarityTemplateParser(max_vocab=max_vocab)
    tids: list[int] = []
    labels: list[int] = []
    n = 0
    bytes_read = 0
    files_used = 0
    day_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    source_bytes: dict[str, int] = defaultdict(int)
    for zpath, member, size in members:
        if n >= max_lines:
            break
        day = zpath.parent.name
        source = next(
            (key for key in ("careerhub", "courses", "portal", "aca", "thuvien", "www", "forti")
             if key in (str(zpath) + "/" + member).lower()),
            "other",
        )
        if day_counts[day] >= per_day_cap:
            continue
        files_used += 1
        local = 0
        print(f"  • {day}/{zpath.name} :: {Path(member).name} ({size/1e6:.1f} MB)")
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                with zf.open(member, "r") as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    for line in text:
                        line = line.strip()
                        if not line:
                            continue
                        tids.append(parser.parse(line))
                        labels.append(int(is_anomaly_line(line)))
                        n += 1
                        local += 1
                        day_counts[day] += 1
                        line_bytes = len(line.encode("utf-8", errors="replace")) + 1
                        bytes_read += line_bytes
                        source_counts[source] += 1
                        source_bytes[source] += line_bytes
                        if (
                            n >= max_lines
                            or local >= per_file_cap
                            or day_counts[day] >= per_day_cap
                        ):
                            break
        except Exception as e:
            print(f"    [!] {e}")
        time.sleep(0.05)
    meta = {
        "lines": n,
        "bytes_read": bytes_read,
        "approx_gb_streamed": round(bytes_read / 1e9, 4),
        "files_used": files_used,
        "vocab_size": len(parser.template_to_id) + 2,
        "line_anomaly_rate": float(np.mean(labels)) if labels else 0.0,
        "lines_per_day": dict(day_counts),
        "lines_per_source": dict(source_counts),
        "bytes_per_source": dict(source_bytes),
    }
    return np.asarray(tids, dtype=np.int32), np.asarray(labels, dtype=np.int8), parser, meta


def stream_partition_windows(tids, labels, window=20, step=1, train_r=0.8, val_r=0.1):
    n = len(tids)
    i_train = int(n * train_r)
    i_val = int(n * (train_r + val_r))
    parts = {
        "train": (tids[:i_train], labels[:i_train]),
        "val": (tids[i_train:i_val], labels[i_train:i_val]),
        "test": (tids[i_val:], labels[i_val:]),
    }
    out = {}
    for name, (tok, lab) in parts.items():
        seqs, seq_y = [], []
        if len(tok) < window:
            out[name] = (np.zeros((0, window), dtype=np.int32), np.zeros((0,), dtype=np.int8))
            continue
        for i in range(0, len(tok) - window + 1, step):
            seqs.append(tok[i : i + window])
            seq_y.append(int(lab[i : i + window].any()))
        out[name] = (np.asarray(seqs, dtype=np.int32), np.asarray(seq_y, dtype=np.int8))
    return out


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(model, X_norm, vocab_size, epochs, batch_size, lr, mask_ratio):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    history = {"train_loss": []}
    n = len(X_norm)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n)
        total, steps = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = torch.tensor(X_norm[idx.numpy()], dtype=torch.long, device=DEVICE)
            opt.zero_grad()
            masked, selected = make_masked_inputs(
                batch, model.mask_token_id, mask_ratio=mask_ratio
            )
            loss = masked_token_loss(model(masked), batch, selected)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        history["train_loss"].append(total / max(steps, 1))
        print(f"    epoch {ep}/{epochs} loss={history['train_loss'][-1]:.4f}")
    return time.time() - t0, history


@torch.no_grad()
def score_sequences(model, X, batch_size=512):
    model.eval()
    scores = []
    t0 = time.perf_counter()
    for i in range(0, len(X), batch_size):
        batch = torch.tensor(X[i : i + batch_size], dtype=torch.long, device=DEVICE)
        scores.append(
            pseudo_log_likelihood_scores(model, batch, model.mask_token_id).cpu().numpy()
        )
    elapsed = time.perf_counter() - t0
    return np.concatenate(scores), (elapsed / max(len(X), 1)) * 1000.0


def best_f1_threshold(scores, y):
    if y.sum() == 0:
        return float(np.percentile(scores, 95)), 0.0
    p, r, thr = precision_recall_curve(y, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    i = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]), float(f1[i])


def nominal_mu_sigma_threshold(scores, y, k: float = 3.0):
    """Paper protocol: τ = μ_val + kσ_val on nominal validation sequences."""
    nominal = scores[y == 0]
    if len(nominal) < 10:
        nominal = scores
    mu = float(np.mean(nominal))
    sigma = float(np.std(nominal))
    return mu + k * sigma, {"mu": mu, "sigma": sigma, "k": k, "n_nominal": int(len(nominal))}


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


def run_seeds(parts, vocab_size, seq_len, epochs, batch_size, mask_ratio):
    X_tr, y_tr = parts["train"]
    X_va, y_va = parts["val"]
    X_te, y_te = parts["test"]
    X_norm = X_tr[y_tr == 0]
    if len(X_norm) < 100:
        X_norm = X_tr
    print(f"  nominal train seqs: {len(X_norm):,} / {len(X_tr):,}")

    seed_rows = []
    loss_curves = []
    last_scores = None
    for seed in SEEDS:
        print(f"\n  === seed {seed} ===")
        set_seed(seed)
        model = TCNTransformerAutoencoder(
            vocab_size=vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, seq_len=seq_len
        ).to(DEVICE)
        train_s, hist = train_one(model, X_norm, vocab_size, epochs, batch_size, 1e-3, mask_ratio)
        loss_curves.append(hist["train_loss"])
        val_scores, _ = score_sequences(model, X_va)
        thr, thr_meta = nominal_mu_sigma_threshold(val_scores, y_va, k=3.0)
        # Diagnostic only (not used for decisions): oracle best-F1 on val
        oracle_thr, oracle_f1 = best_f1_threshold(val_scores, y_va)
        te_scores, lat = score_sequences(model, X_te)
        m = metrics_at(te_scores, y_te, thr)
        row = {
            "seed": seed,
            "threshold": thr,
            "threshold_meta": thr_meta,
            "oracle_val_best_f1_threshold": oracle_thr,
            "oracle_val_best_f1": oracle_f1,
            "latency_ms_per_seq": lat,
            "train_seconds": train_s,
            **m,
        }
        seed_rows.append(row)
        last_scores = (te_scores, y_te, thr)
        print(
            f"    test F1={m['f1']:.4f} AUC={m['roc_auc']:.4f} "
            f"P={m['precision']:.4f} R={m['recall']:.4f} lat={lat:.4f}ms"
        )

    def agg(key):
        vals = [r[key] for r in seed_rows if r.get(key) is not None and not np.isnan(r[key])]
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": vals,
        }

    summary = {
        "precision": agg("precision"),
        "recall": agg("recall"),
        "f1": agg("f1"),
        "roc_auc": agg("roc_auc"),
        "auprc": agg("auprc"),
        "latency_ms_per_seq": agg("latency_ms_per_seq"),
        "per_seed": seed_rows,
        "mean_train_loss_curve": list(np.mean(np.asarray(loss_curves), axis=0)),
    }

    # ROC/PR for last seed (representative for figures)
    te_scores, y_te, thr = last_scores
    fpr, tpr, _ = roc_curve(y_te, te_scores)
    prec, rec, _ = precision_recall_curve(y_te, te_scores)
    summary["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    summary["pr_curve"] = {"precision": prec.tolist(), "recall": rec.tolist()}
    summary["final_threshold"] = thr
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"D:\huflit-campus-logs")
    ap.add_argument("--max_lines", type=int, default=1_000_000)
    ap.add_argument("--per_file_cap", type=int, default=40_000)
    ap.add_argument("--per_day_cap", type=int, default=200_000)
    ap.add_argument("--max_member_mb", type=float, default=500.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_vocab", type=int, default=2500)
    ap.add_argument("--mask_ratio", type=float, default=0.15)
    ap.add_argument(
        "--include_fortigate",
        action="store_true",
        help="Opt in to FortiGate ZIP members; disabled by default to avoid very high I/O.",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = ROOT / "data" / "processed" / "huflit"
    res_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" SOICT 2026 — HUFLIT STREAM-POSITION TRAIN (proxy labels / multi-seed)")
    print(f" Device: {DEVICE}")
    print(f" Data : {data_root}")
    print("=" * 72)

    members = discover_log_members(data_root, args.max_member_mb, args.include_fortigate)
    print(f"Eligible .log members: {len(members)}")
    if not members:
        raise SystemExit("No .log members found. Check selective extract.")

    t0 = time.time()
    tids, labels, parser, meta = build_token_stream(
        members, args.max_lines, args.per_file_cap, args.max_vocab, args.per_day_cap
    )
    print(f"Streamed {meta['lines']:,} lines (~{meta['approx_gb_streamed']} GB text) in {time.time()-t0:.1f}s")
    print(f"Line anomaly rate (proxy labels): {meta['line_anomaly_rate']*100:.2f}%")

    parts = stream_partition_windows(tids, labels, args.window, args.step)
    for k, (X, y) in parts.items():
        print(f"  {k:5s}: {len(X):,} seqs | anomaly {y.mean()*100:.2f}%")

    # Persist arrays for reuse
    np.save(out_dir / "train_sequences.npy", parts["train"][0])
    np.save(out_dir / "train_labels.npy", parts["train"][1])
    np.save(out_dir / "val_sequences.npy", parts["val"][0])
    np.save(out_dir / "val_labels.npy", parts["val"][1])
    np.save(out_dir / "test_sequences.npy", parts["test"][0])
    np.save(out_dir / "test_labels.npy", parts["test"][1])
    with open(out_dir / "templates.json", "w", encoding="utf-8") as f:
        json.dump(parser.template_to_id, f, ensure_ascii=False)

    vocab_size = max(parser.template_to_id.values()) + 2
    summary = run_seeds(
        parts, vocab_size, args.window, args.epochs, args.batch_size, args.mask_ratio
    )

    payload = {
        "experiment": "harden_huflit_train",
        "device": str(DEVICE),
        "data_root": str(data_root),
        "corpus_note": (
            "Selective extract from Thang6 rar (~1.43 GB zip payloads on disk); "
            "streamed .log members only; FortiGate excluded by default; skipped members > "
            f"{args.max_member_mb} MB. NOT the full 21 GB uncompressed archive."
        ),
        "labeling": (
            "Proxy labels: HTTP 4xx/5xx, FortiGate deny/block/attack fields, "
            "and keyword heuristics. Not human gold labels."
        ),
        "protocol": {
            "split": "stream-position 80/10/10 before windowing; repeated-backup overlap must be audited",
            "window": args.window,
            "step": args.step,
            "train": "masked CE on nominal sequences only",
            "mask_ratio": args.mask_ratio,
            "threshold": "tau = mu_val + 3*sigma_val on nominal validation sequences",
            "seeds": list(SEEDS),
            "epochs": args.epochs,
            "per_day_cap": args.per_day_cap,
            "per_file_cap": args.per_file_cap,
            "max_lines": args.max_lines,
            "max_member_mb": args.max_member_mb,
            "include_fortigate": args.include_fortigate,
            "parser": "custom similarity template parser (not Drain3)",
            "scoring": "leave-one-position-out pseudo-log-likelihood",
        },
        "ingest_meta": meta,
        "partition_sizes": {k: {"n": int(len(v[0])), "anomaly_rate": float(v[1].mean()) if len(v[1]) else 0.0} for k, v in parts.items()},
        "metrics": summary,
        "headline": {
            "precision": summary["precision"]["mean"],
            "precision_std": summary["precision"]["std"],
            "recall": summary["recall"]["mean"],
            "recall_std": summary["recall"]["std"],
            "f1": summary["f1"]["mean"],
            "f1_std": summary["f1"]["std"],
            "roc_auc": summary["roc_auc"]["mean"],
            "roc_auc_std": summary["roc_auc"]["std"],
            "auprc": summary["auprc"]["mean"],
            "latency_ms_per_seq": summary["latency_ms_per_seq"]["mean"],
        },
    }

    out_json = res_dir / "huflit_hardened_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    # Also overwrite evaluation_summary with honest numbers
    with open(res_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": "HUFLIT Campus Logs (selective Thang6 stream)",
                "total_test_sequences": payload["partition_sizes"]["test"]["n"],
                "precision": payload["headline"]["precision"],
                "recall": payload["headline"]["recall"],
                "f1_score": payload["headline"]["f1"],
                "roc_auc": payload["headline"]["roc_auc"],
                "auprc": payload["headline"]["auprc"],
                "latency_ms_per_seq": payload["headline"]["latency_ms_per_seq"],
                "source": str(out_json),
            },
            f,
            indent=2,
        )

    h = payload["headline"]
    print("\n" + "=" * 72)
    print(" HARDENED HUFLIT RESULTS (mean ± std over 5 seeds)")
    print(
        f"  F1={h['f1']:.4f}±{h['f1_std']:.4f}  "
        f"AUC={h['roc_auc']:.4f}±{h['roc_auc_std']:.4f}  "
        f"P={h['precision']:.4f}  R={h['recall']:.4f}  "
        f"lat={h['latency_ms_per_seq']:.4f} ms"
    )
    print(f"  Saved → {out_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

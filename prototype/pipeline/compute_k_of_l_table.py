"""
Correct Table 6: k-of-L aggregation sensitivity on reconstructed raw-line
proxy labels (fixed PCA scores; seed 42). Also optional trigger-visibility
audit against stored OR labels (not a substitute for k-of-L).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "huflit_v2"
LLDIR = ROOT / "data" / "private" / "huflit_v2_line_labels"
OUT = ROOT / "results" / "tables" / "k_of_l_sensitivity.json"

TRIGGER = re.compile(
    r"(?:\b(?:4\d{2}|5\d{2})\b|deny|block|error|fail|critical|attack|unauthorized|"
    r"forbidden|refused|invalid|malware|intrusion|alert|drop|reject)",
    re.I,
)


def always_anomaly_f1(y: np.ndarray) -> float:
    p = float(np.mean(y)) if len(y) else 0.0
    return float(2 * p / (1 + p)) if p > 0 else 0.0


def aggregate(line_labels: np.ndarray, k: int) -> np.ndarray:
    return (line_labels.sum(axis=1) >= k).astype(np.int64)


def count_vectorize(X: np.ndarray, vocab: int) -> np.ndarray:
    out = np.zeros((len(X), vocab), dtype=np.float32)
    for i, row in enumerate(X):
        for t in row:
            t = int(t)
            if 0 < t < vocab:
                out[i, t] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out / n


def pca_test_scores(X_tr, y_tr_or, X_te, vocab: int, seed: int = 42) -> np.ndarray:
    """Fit PCA once on OR-nominal train; return continuous scores for test windows."""
    Xtr = count_vectorize(X_tr, vocab)
    Xte = count_vectorize(X_te, vocab)
    normal = y_tr_or == 0
    n_comp = max(1, min(32, int(normal.sum()) - 1, vocab - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(Xtr[normal])
    recon = pca.inverse_transform(pca.transform(Xte))
    return np.mean((Xte - recon) ** 2, axis=1)


def metrics_for_labels(scores: np.ndarray, y: np.ndarray) -> dict:
    y = y.astype(np.int64)
    prev = float(y.mean()) if len(y) else 0.0
    if y.sum() == 0 or y.sum() == len(y):
        return {
            "n_pos": int(y.sum()),
            "prevalence": prev,
            "always_anomaly_f1": always_anomaly_f1(y),
            "pca_roc_auc": float("nan"),
            "pca_auprc": float("nan"),
            "pca_auprc_lift": float("nan"),
        }
    auprc = float(average_precision_score(y, scores))
    return {
        "n_pos": int(y.sum()),
        "prevalence": prev,
        "always_anomaly_f1": always_anomaly_f1(y),
        "pca_roc_auc": float(roc_auc_score(y, scores)),
        "pca_auprc": auprc,
        "pca_auprc_lift": float(auprc - prev),
    }


def trigger_visibility_audit(X_te, y_or, id_to_str: dict) -> dict:
    triggers = {i for i, s in id_to_str.items() if TRIGGER.search(str(s) or "")}
    pred = np.array(
        [1 if any(int(t) in triggers for t in row) else 0 for row in X_te],
        dtype=np.int64,
    )
    y = y_or.astype(np.int64)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n_trigger_templates": len(triggers),
        "pred_positive_rate": float(pred.mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": cm.ravel().tolist(),
        "trigger_rate_among_stored_positives": float(pred[y == 1].mean()) if (y == 1).any() else 0.0,
        "trigger_rate_among_stored_negatives": float(pred[y == 0].mean()) if (y == 0).any() else 0.0,
        "note": (
            "Predicts positive if any visible template matches keyword/status triggers; "
            "target is stored OR proxy labels. Measures observable proxy-related features, "
            "not SOC ground truth and not equivalent to k-of-L aggregation."
        ),
    }


def main():
    man = json.loads((LLDIR / "manifest.json").read_text(encoding="utf-8"))
    if man.get("status") != "OK":
        raise SystemExit("Line-label reconstruction not OK; aborting.")

    ll_te = np.load(LLDIR / "test_line_labels.npy")
    ll_tr = np.load(LLDIR / "train_line_labels.npy")
    y_te_or = np.load(DATA / "test_labels.npy").astype(np.int64)
    y_tr_or = np.load(DATA / "train_labels.npy").astype(np.int64)
    X_te = np.load(DATA / "test_sequences.npy")
    X_tr = np.load(DATA / "train_sequences.npy")
    with open(DATA / "templates.json", encoding="utf-8") as f:
        tmpl = json.load(f)
    id_to_str = {int(v): k for k, v in tmpl.items()} if tmpl and isinstance(next(iter(tmpl.values())), int) else {
        int(k): v for k, v in tmpl.items()
    }
    vocab = max(id_to_str.keys(), default=0) + 2

    assert ll_te.shape == (19990, 20)
    y1 = aggregate(ll_te, 1)
    assert np.array_equal(y1, y_te_or), "k=1 must equal stored OR"
    assert np.all(aggregate(ll_te, 3) <= aggregate(ll_te, 2))
    assert np.all(aggregate(ll_te, 2) <= aggregate(ll_te, 1))

    scores = pca_test_scores(X_tr, y_tr_or, X_te, vocab, seed=42)

    rows = []
    for k, name in [(1, "1-of-L (= OR)"), (2, "2-of-L"), (3, "3-of-L")]:
        y = aggregate(ll_te, k)
        m = metrics_for_labels(scores, y)
        m["aggregation"] = name
        m["k"] = k
        rows.append(m)
        print(name, m, flush=True)

    # Monotonic prevalence check
    assert rows[0]["prevalence"] >= rows[1]["prevalence"] >= rows[2]["prevalence"]

    visibility = trigger_visibility_audit(X_te, y_te_or, id_to_str)
    print("visibility", visibility, flush=True)

    payload = {
        "experiment": "protocol_b_k_of_l_fixed_pca_seed42",
        "seed": 42,
        "pca_fit": "nominal train under stored OR labels; scores fixed across k",
        "line_label_source": str(LLDIR / "test_line_labels.npy"),
        "reconstruction_status": man["status"],
        "rows": rows,
        "trigger_visibility_audit": visibility,
        "commands": [
            "python prototype/pipeline/reconstruct_line_labels.py",
            "python prototype/pipeline/compute_k_of_l_table.py",
            "python prototype/pipeline/test_label_aggregation.py",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""
Camera-ready extras (SOICT #5848): label-rule aggregation sensitivity + PCA under
k-of-L window labels. Uses only data/processed/{huflit,huflit_v2} — no RAR extract.

Proxy line positives inside a window are approximated by templates whose string
matches the same keyword/status patterns used in the paper's proxy rules
(HTTP 4xx/5xx, deny/block/error-like). This is a rule-consistency / prevalence
sensitivity study, not a claim of SOC ground truth.
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
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "camera_ready_extras.json"

TRIGGER = re.compile(
    r"(?:\b(?:4\d{2}|5\d{2})\b|deny|block|error|fail|critical|attack|unauthorized|"
    r"forbidden|refused|invalid|malware|intrusion|alert|drop|reject)",
    re.I,
)


def load_protocol(name: str):
    data = ROOT / "data" / "processed" / name
    X_te = np.load(data / "test_sequences.npy")
    y_or = np.load(data / "test_labels.npy").astype(np.int8)
    X_tr = np.load(data / "train_sequences.npy")
    y_tr = np.load(data / "train_labels.npy").astype(np.int8)
    with open(data / "templates.json", encoding="utf-8") as f:
        tmpl = json.load(f)
    # templates.json may be {template: id} or id-keyed
    if tmpl and isinstance(next(iter(tmpl.values())), int):
        id_to_str = {int(v): k for k, v in tmpl.items()}
    else:
        id_to_str = {int(k): v for k, v in tmpl.items()}
    return X_tr, y_tr, X_te, y_or, id_to_str


def trigger_ids(id_to_str: dict) -> set[int]:
    return {i for i, s in id_to_str.items() if TRIGGER.search(str(s) or "")}


def line_proxy_counts(X: np.ndarray, triggers: set[int]) -> np.ndarray:
    # count of trigger-template positions per window
    return np.array([sum(1 for t in row if int(t) in triggers) for row in X], dtype=np.int16)


def label_at_k(counts: np.ndarray, k: int) -> np.ndarray:
    return (counts >= k).astype(np.int8)


def always_anomaly_f1(y: np.ndarray) -> float:
    p = float(y.mean()) if len(y) else 0.0
    return float(2 * p / (1 + p)) if p > 0 else 0.0


def count_vectorize(X: np.ndarray, vocab: int) -> np.ndarray:
    out = np.zeros((len(X), vocab), dtype=np.float32)
    for i, row in enumerate(X):
        for t in row:
            t = int(t)
            if 0 < t < vocab:
                out[i, t] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
    return out / n


def pca_rank_metrics(X_tr, y_tr, X_te, y_te, vocab: int, seed: int = 42):
    if y_te.sum() == 0 or y_te.sum() == len(y_te):
        return {"roc_auc": float("nan"), "auprc": float("nan"), "lift": float("nan"), "f1_3sigma": float("nan")}
    Xtr = count_vectorize(X_tr, vocab)
    Xte = count_vectorize(X_te, vocab)
    normal = y_tr == 0
    if normal.sum() < 10:
        return {"roc_auc": float("nan"), "auprc": float("nan"), "lift": float("nan"), "f1_3sigma": float("nan")}
    n_comp = max(1, min(32, Xtr.shape[1], int(normal.sum()) - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(Xtr[normal])
    recon_n = pca.inverse_transform(pca.transform(Xtr[normal]))
    train_err = np.mean((Xtr[normal] - recon_n) ** 2, axis=1)
    mu, sig = float(train_err.mean()), float(train_err.std() + 1e-12)
    thr = mu + 3 * sig
    recon_t = pca.inverse_transform(pca.transform(Xte))
    test_err = np.mean((Xte - recon_t) ** 2, axis=1)
    y_pred = (test_err > thr).astype(int)
    prev = float(y_te.mean())
    auprc = float(average_precision_score(y_te, test_err))
    return {
        "roc_auc": float(roc_auc_score(y_te, test_err)),
        "auprc": auprc,
        "lift": float(auprc - prev),
        "f1_3sigma": float(f1_score(y_te, y_pred, zero_division=0)),
        "prevalence": prev,
        "always_anomaly_f1": always_anomaly_f1(y_te),
    }


def eval_protocol(name: str):
    X_tr, y_tr_or, X_te, y_te_or, id_to_str = load_protocol(name)
    triggers = trigger_ids(id_to_str)
    vocab = max(id_to_str.keys(), default=0) + 2
    c_tr = line_proxy_counts(X_tr, triggers)
    c_te = line_proxy_counts(X_te, triggers)

    rows = []
    # OR from stored labels (authoritative for paper Protocol A/B)
    rows.append(
        {
            "aggregation": "OR (stored proxy window labels)",
            "k": 1,
            "prevalence": float(y_te_or.mean()),
            "always_anomaly_f1": always_anomaly_f1(y_te_or),
            "note": "authoritative labels used in main tables",
            "pca": pca_rank_metrics(X_tr, y_tr_or, X_te, y_te_or, vocab),
        }
    )
    # Template-trigger k-of-L sensitivity (same patterns as proxy keywords)
    for k, tag in [(1, "1-of-L trigger-approx"), (2, "2-of-L trigger-approx"), (3, "3-of-L trigger-approx")]:
        y_tr = label_at_k(c_tr, k)
        y_te = label_at_k(c_te, k)
        # Rule oracle: predict positive iff count>=k — F1=1 by construction vs these labels
        rows.append(
            {
                "aggregation": tag,
                "k": k,
                "prevalence": float(y_te.mean()),
                "always_anomaly_f1": always_anomaly_f1(y_te),
                "rule_oracle_f1": 1.0,
                "rule_oracle_note": (
                    "Applying the same count>=k rule that defines the labels yields F1=1; "
                    "this is a proxy-label consistency check, not an independent detector."
                ),
                "pca": pca_rank_metrics(X_tr, y_tr, X_te, y_te, vocab),
                "trigger_templates": len(triggers),
            }
        )
    return {
        "protocol": name,
        "n_test": int(len(X_te)),
        "n_train": int(len(X_tr)),
        "L": int(X_te.shape[1]) if len(X_te) else 0,
        "rows": rows,
    }


def main():
    payload = {
        "purpose": "SOICT camera-ready label-aggregation sensitivity",
        "protocols": [eval_protocol("huflit"), eval_protocol("huflit_v2")],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()

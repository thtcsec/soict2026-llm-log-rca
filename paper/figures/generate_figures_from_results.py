"""Regenerate detector figures from the canonical measured baseline JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = json.loads((ROOT / "results" / "tables" / "huflit_baselines.json").read_text(encoding="utf-8"))
OUT = Path(__file__).resolve().parent
PROPOSED = RES["models"]["Proposed Masked TCN-Transformer"]


def fig_loss():
    curve = PROPOSED["mean_train_loss_curve"]
    epochs = np.arange(1, len(curve) + 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(epochs, curve, "o-", color="#1f77b4", lw=1.8, label="Masked train loss (mean@5 seeds)")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Masked Cross-Entropy Loss")
    ax.set_title("Masked-Token Loss Convergence", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tcn_loss.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_roc_pr():
    diag = RES["diagnostics"]["proposed"]
    roc = diag["roc_curve"]
    pr = diag["pr_curve"]
    h = diag["representative_metrics"]
    seed = diag["representative_seed"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.0))
    ax1.plot(roc["fpr"], roc["tpr"], color="#2ca02c", lw=1.6, label=f"Masked TCN–Transformer (AUC={h['roc_auc']:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC (seed {seed})", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=7.0)

    ax2.plot(pr["recall"], pr["precision"], color="#2ca02c", lw=1.6, label=f"Masked TCN–Transformer (AP={h['auprc']:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall (seed {seed})", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower left", fontsize=7.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_roc_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def fig_latency():
    # Prefer Protocol B validity JSON when present (No-TCN vs TCN batch-1).
    v2_path = ROOT / "results" / "tables" / "huflit_v2_baselines.json"
    if v2_path.exists():
        v2 = json.loads(v2_path.read_text(encoding="utf-8"))
        no = v2["models"]["Masked Transformer (No TCN)"]["primary_mu3sigma"]["batch1_ms"]["mean"]
        tcn = v2["models"]["Proposed Masked TCN-Transformer"]["primary_mu3sigma"]["batch1_ms"]["mean"]
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        labels = ["Masked Transformer\n(No TCN)", "Masked TCN-\nTransformer"]
        vals = [no, tcn]
        ax.bar(labels, vals, color=["#1f77b4", "#d62728"], width=0.55, edgecolor="black")
        ax.set_ylabel("Batch-1 PLL latency (ms)")
        ax.set_title("Protocol B: TCN adds scoring latency", fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(True, axis="y", linestyle=":", alpha=0.5)
        for i, value in enumerate(vals):
            ax.text(i, value * 1.05, f"{value:.2f}", ha="center", fontsize=10, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / "fig4_batch1_pll_latency.png", dpi=300, bbox_inches="tight")
        # Keep legacy filename in sync for older scripts.
        fig.savefig(OUT / "fig4_llm_latency_triage.png", dpi=300, bbox_inches="tight")
        plt.close()
        return
    batch1 = PROPOSED["batch1_latency_ms"]["mean"]
    batched = PROPOSED["batched_ms_per_seq"]["mean"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = ["Batch=1\nwall clock", "Batch=512\nms/sequence"]
    vals = [batch1, batched]
    ax.bar(labels, vals, color=["#2ca02c", "#1f77b4"], width=0.55, edgecolor="black")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("Milliseconds")
    ax.set_title("Measured CPU Scoring Latency", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    for i, value in enumerate(vals):
        ax.text(i, value * 1.04, f"{value:.3f}", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_llm_latency_triage.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    fig_loss()
    fig_roc_pr()
    fig_latency()
    print("[+] Regenerated fig2/fig3/fig4 from measured results")

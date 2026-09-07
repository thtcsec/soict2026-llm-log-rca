"""
generate_all_soict_figures.py - Generates 4 High-Resolution (300 DPI) Graphics for SOICT 2026 Paper
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

def generate_fig1_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=300)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def draw_card(ax, x, y, w, h, title, subtitle, bg_color="#ffffff", edge_color="#333333", title_color="#000000", sub_color="#444444", lw=1.2, radius=1.0):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                                    facecolor=bg_color, edgecolor=edge_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color=title_color, zorder=3)
            ax.text(x + w/2, y + h*0.28, subtitle, ha="center", va="center", fontsize=7.6, color=sub_color, zorder=3)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=9.0, fontweight="bold", color=title_color, zorder=3)

    def draw_stage(ax, x, y, w, h, title, bg_color, header_color):
        stage_box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5",
                                           facecolor=bg_color, edgecolor=header_color, linewidth=1.8, linestyle="--", zorder=1)
        ax.add_patch(stage_box)
        header_box = patches.FancyBboxPatch((x + 1.0, y + h - 4.2), w - 2.0, 3.4, boxstyle="round,pad=0.1,rounding_size=0.8",
                                            facecolor=header_color, edgecolor="none", zorder=2)
        ax.add_patch(header_box)
        ax.text(x + w/2, y + h - 2.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=3)

    # Stage 1: Ingestion & Parsing
    draw_stage(ax, 2, 2, 28, 38, "Stage 1: Streaming Parsing", "#f0f7ff", "#1f77b4")
    draw_card(ax, 4, 27.5, 24, 6.0, "Selective Campus Stream", "1,000,000 lines / five backup days", "#ffffff", "#1f77b4", "#1f77b4")
    draw_card(ax, 4, 16.5, 24, 6.0, "Similarity Template Parser", "Regex masking; custom parser, not Drain3", "#ffffff", "#1f77b4", "#1f77b4")
    draw_card(ax, 4, 5.5, 24, 6.0, "Template Sequence Vector", "X = (e1, e2, ..., eL) in V^L", "#ffffff", "#1f77b4", "#1f77b4")

    # Intra-Stage 1 Arrows
    ax.annotate("", xy=(16, 22.8), xytext=(16, 27.2), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.5, color="#1f77b4", zorder=4))
    ax.annotate("", xy=(16, 11.8), xytext=(16, 16.2), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.5, color="#1f77b4", zorder=4))

    # Inter-Stage Arrow 1 -> 2
    ax.annotate("", xy=(34, 8.5), xytext=(28.2, 8.5), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#2c3e50", zorder=5))

    # Stage 2: Masked Sequence Anomaly Detection
    draw_stage(ax, 34, 2, 33, 38, "Stage 2: Masked Sequence Detection", "#f2faf2", "#2ca02c")
    draw_card(ax, 36, 27.5, 29, 6.0, "Dynamic 15% Token Masking", "Input Embedding + [MASK] Tokens", "#ffffff", "#2ca02c", "#2ca02c")
    draw_card(ax, 36, 19.5, 29, 5.5, "Dilated Causal 1D TCN", "Local Temporal Patterns (d=1,2)", "#ffffff", "#2ca02c", "#2ca02c")
    draw_card(ax, 36, 12.0, 29, 5.5, "Transformer Encoder", "Multi-Head Self-Attention + Pre-LN", "#ffffff", "#2ca02c", "#2ca02c")
    draw_card(ax, 36, 4.5, 29, 5.5, "Leave-One-Out PLL Scoring", "Score S(X) > calibrated threshold", "#2ca02c", "#2ca02c", "#ffffff", "#e8f5e9")

    # Intra-Stage 2 Arrows
    ax.annotate("", xy=(50.5, 25.2), xytext=(50.5, 27.2), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#2ca02c", zorder=4))
    ax.annotate("", xy=(50.5, 17.7), xytext=(50.5, 19.2), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#2ca02c", zorder=4))
    ax.annotate("", xy=(50.5, 10.2), xytext=(50.5, 11.7), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#2ca02c", zorder=4))

    # Inter-Stage Arrow 2 -> 3
    ax.annotate("", xy=(71, 7.2), xytext=(67.2, 7.2), arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#d62728", zorder=5))

    # Stage 3: RAG-Grounded Foundation Model RCA
    draw_stage(ax, 71, 2, 32, 38, "Stage 3: RCA Design (Unevaluated)", "#fff5f5", "#d62728")
    draw_card(ax, 73, 27.5, 28, 6.0, "Institutional SOP Repository", "BGE + FAISS (design; unevaluated)", "#ffffff", "#d62728", "#d62728")
    draw_card(ax, 73, 19.5, 28, 5.5, "Context Prompt Assembly", "Telemetry + Retrieved SOP Guidance", "#ffffff", "#d62728", "#d62728")
    draw_card(ax, 73, 12.0, 28, 5.5, "4-bit Foundation Model", "Phi-3 Mini (NF4) / Mistral (AWQ INT4)", "#ffffff", "#d62728", "#d62728")
    draw_card(ax, 73, 4.5, 28, 5.5, "Structured Incident Playbook", "Diagnosis, Root Cause & Mitigation", "#d62728", "#d62728", "#ffffff", "#ffebee")

    # Intra-Stage 3 Arrows
    ax.annotate("", xy=(87, 25.2), xytext=(87, 27.2), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#d62728", zorder=4))
    ax.annotate("", xy=(87, 17.7), xytext=(87, 19.2), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#d62728", zorder=4))
    ax.annotate("", xy=(87, 10.2), xytext=(87, 11.7), arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#d62728", zorder=4))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    generate_fig1_architecture()
    from generate_figures_from_results import fig_loss, fig_roc_pr, fig_latency
    fig_loss()
    fig_roc_pr()
    fig_latency()
    print("[+] Generated architecture plus measured detector figures only.")

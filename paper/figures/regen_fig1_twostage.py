"""Regenerate Fig.1 as a two-stage measured pipeline (no RCA stage)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))

def generate_fig1_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def draw_card(ax, x, y, w, h, title, subtitle, edge, title_c):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0",
            facecolor="#ffffff", edgecolor=edge, linewidth=1.2, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=title_c, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=8, color="#444444", zorder=3)

    def draw_stage(ax, x, y, w, h, title, bg, header):
        stage = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.5",
            facecolor=bg, edgecolor=header, linewidth=1.8, linestyle="--", zorder=1,
        )
        ax.add_patch(stage)
        header_box = patches.FancyBboxPatch(
            (x + 1.2, y + h - 4.0), w - 2.4, 3.2,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor=header, edgecolor="none", zorder=2,
        )
        ax.add_patch(header_box)
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=3)

    # Stage 1
    draw_stage(ax, 3, 2, 42, 36, "Stage 1: Streaming Parsing", "#f0f7ff", "#1f77b4")
    draw_card(ax, 6, 26.5, 36, 5.5, "Selective Campus Stream",
              "1,000,000 deduplicated lines / five backup days", "#1f77b4", "#1f77b4")
    draw_card(ax, 6, 16.5, 36, 5.5, "Similarity Template Parser",
              "Regex masking; custom parser (not Drain3)", "#1f77b4", "#1f77b4")
    draw_card(ax, 6, 6.5, 36, 5.5, "Source-bounded Template Windows",
              "L=20, stride=10; no cross-member windows (Protocol B)", "#1f77b4", "#1f77b4")
    ax.annotate("", xy=(24, 22.2), xytext=(24, 26.2),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.5, color="#1f77b4"))
    ax.annotate("", xy=(24, 12.2), xytext=(24, 16.2),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.5, color="#1f77b4"))

    ax.annotate("", xy=(50, 9), xytext=(45.5, 9),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#2c3e50"))

    # Stage 2
    draw_stage(ax, 50, 2, 47, 36, "Stage 2: Masked Sequence Detection", "#f2faf2", "#2ca02c")
    draw_card(ax, 53, 26.5, 41, 5.5, "Dynamic 15% Token Masking",
              "True [MASK] replacement + sinusoidal positions", "#2ca02c", "#2ca02c")
    draw_card(ax, 53, 18.5, 41, 5.0, "Optional Causal TCN + Transformer",
              "Dilations {1,2}; 1 encoder layer; d_model=128", "#2ca02c", "#2ca02c")
    draw_card(ax, 53, 10.5, 41, 5.0, "Leave-One-Out PLL Scoring",
              "S(X); threshold mu+k sigma / percentiles", "#2ca02c", "#2ca02c")
    draw_card(ax, 53, 4.0, 41, 4.5, "Validity Metrics",
              "AUC, AUPRC lift, always-anomaly baseline", "#2ca02c", "#ffffff")
    ax.annotate("", xy=(73.5, 23.7), xytext=(73.5, 26.2),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#2ca02c"))
    ax.annotate("", xy=(73.5, 15.7), xytext=(73.5, 18.2),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#2ca02c"))
    ax.annotate("", xy=(73.5, 8.7), xytext=(73.5, 10.2),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", lw=1.4, color="#2ca02c"))

    out = os.path.join(OUTPUT_DIR, "fig1_architecture.png")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[+] wrote", out)


if __name__ == "__main__":
    generate_fig1_architecture()

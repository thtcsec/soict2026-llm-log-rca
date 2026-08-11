"""
generate_paper_figures.py - Generates 300 DPI graphics for SOICT 2026 Paper
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8
})

def generate_fig1():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    
    boxes = [
        ("Raw 5GB Log Stream\n(HUFLIT / BGL)", 0.1, 0.5, "#1f77b4"),
        ("Drain3 Online\nLog Parser", 0.32, 0.5, "#ff7f0e"),
        ("TCN-Transformer\nAutoencoder", 0.54, 0.5, "#2ca02c"),
        ("LLM RCA Narration\n(Phi-3 / Mistral)", 0.76, 0.5, "#d62728")
    ]
    
    for label, x, y, color in boxes:
        ax.text(x, y, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.85, edgecolor="black"), color="white", fontweight="bold", fontsize=8)
        
    # Arrows
    arrows = [(0.21, 0.32), (0.43, 0.54), (0.65, 0.76)]
    for start, end in arrows:
        ax.annotate("", xy=(end - 0.08, 0.5), xytext=(start + 0.08, 0.5),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
        
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("End-to-End LLM-Augmented AIOps Pipeline Architecture", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_architecture.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_fig1()
    print("[+] Generated fig1_architecture.png in paper/figures/")

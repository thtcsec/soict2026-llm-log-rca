<p align="center">
  <img src="assets/soict-hust.png" height="80" alt="HUST SOICT Logo" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/hcmus.png" height="80" alt="HCMUS VNU-HCM Logo" />
</p>

<h1 align="center">SOICT 2026 — Campus Log Anomaly Detection under Proxy Labels</h1>

<p align="center">
  <a href="https://soict.org"><img src="https://img.shields.io/badge/Conference-SOICT%202026-0066CC.svg" alt="Conference" /></a>
  <a href="https://soict.org"><img src="https://img.shields.io/badge/Organizers-HUST%20%7C%20VNU--HCMUS-CC0000.svg" alt="Organizers" /></a>
  <a href="https://springer.com"><img src="https://img.shields.io/badge/Proceedings-Springer%20CCIS-FF6600.svg" alt="Proceedings" /></a>
  <img src="https://img.shields.io/badge/Indexing-Scopus%20%7C%20EI%20Compendex-6A0DAD.svg" alt="Indexing" />
  <img src="https://img.shields.io/badge/Status-Integrity%20hardening-orange.svg" alt="Status" />
</p>

---

## 📌 Paper Metadata

| Field | Info |
|---|---|
| **Title** | *Campus Log Anomaly Detection under Proxy Labels: A TCN–Transformer Study and Design for Local RCA Narration* |
| **Track** | Software Engineering, Trusted Digital Platforms, and Smart Services (Secondary: Applied AI) |
| **Authors** | Thanh Tien Cao∗, Tu Hoang Trinh, Ha Manh Tran (PGS.TS) |
| **Affiliation** | Faculty of Information Technology, HUFLIT |
| **Artifact** | https://github.com/thtcsec/soict2026-llm-log-rca (**private**) |
| **Measured result (corrected)** | TCN–Transformer F1 **0.0395±0.0163**, AUC **0.7226±0.0178**; masked Transformer without TCN F1 **0.3475±0.4097**, AUC **0.7685±0.0351** |
| **Validity scope** | HUFLIT selective stream: 1,000,000 lines; heuristic proxy labels; stream-position split with documented duplicate-backup overlap |
| **Venue** | Ho Chi Minh City, Vietnam — December 4–5, 2026 |
| **Submission Deadline** | Full Paper: September 16, 2026 |

---

## 💡 Overview

Modern enterprise campus networks generate massive volumes of heterogeneous, unstructured log streams. While deep learning models achieve high accuracy in sequence anomaly detection, traditional AIOps platforms suffer from a **critical explainability bottleneck**: raw anomaly scores fail to communicate actionable root causes, forcing SOC analysts to manually triage thousands of log messages.

This repository evaluates a detector and documents an **unevaluated RCA design**. It does not claim a deployed end-to-end LLM system:

```
[Selective campus stream: 1,000,000 lines]
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Similarity Parser │  ← Custom template extraction (not Drain3)
└─────────────┬───────────────┘
              │ Template ID sequences
              ▼
┌─────────────────────────────┐
│  Stage 2: TCN-Transformer   │  ← Masked-token anomaly detection with PLL
│  Detector                   │
└─────────────┬───────────────┘
              │ Anomaly score > threshold
              ▼
┌─────────────────────────────┐
│  Stage 3: RCA Design Mock   │  ← Prompt/schema only; no measured LLM or RAG
│  (unevaluated)              │
└─────────────────────────────┘
```

---

## 📊 Datasets

| Dataset | Source | Size | Scope |
|---|---|---|---|
| **HUFLIT Campus Logs** | Internal production selective stream | 1,000,000 lines / 368,940,230 streamed bytes | Five backup days; proxy labels; not publicly released |

The current paper does not report BGL, HDFS, or UNSW-NB15 measurements.

---

## 🏗️ Repository Structure

```
soict2026-llm-log-rca/
├── assets/                  # Conference logos (HUST SOICT, HCMUS)
├── data/
│   ├── raw/                 # Raw log files (excluded from Git via .gitignore)
│   └── processed/           # Parsed templates & sequence vectors
├── prototype/
│   ├── drain3/              # Legacy simplified parser (not Drain3 package)
│   ├── pipeline/            # Masked TCN-Transformer detector
│   └── llm/                 # Unevaluated RCA prompt/schema mock
├── results/
│   ├── figures/             # 300 DPI publication graphics
│   └── tables/              # Evaluation CSV metrics
├── paper/
│   ├── soict2026.tex        # Main Springer CCIS LaTeX manuscript
│   ├── soict2026.pdf        # Compiled preview PDF
│   └── figures/             # Embedded paper graphics
├── references/              # Related papers & literature
├── requirements.txt
├── run.sh                   # 1-click pipeline execution
└── soict2026_outline.md     # Detailed research outline & timeline
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place raw logs in data/raw/ (see data/raw/README.md)

# 3. Re-run only measured baselines from existing processed arrays (2 CPU threads)
python prototype/pipeline/run_baselines_huflit.py --threads 2
```

The raw-ingest script excludes every FortiGate member by default. Do not use
`--include_fortigate` on machines where archive I/O causes thermal or disk stress.

---

## 📄 BibTeX Citation

```bibtex
@article{tu2026soict,
  title={Campus Log Anomaly Detection under Proxy Labels: A TCN--Transformer Study
         and Design for Local RCA Narration},
  author={Cao, Thanh Tien and Trinh, Tu Hoang and Tran, Ha Manh},
  journal={Manuscript in preparation for the 15th International Symposium on Information
           and Communication Technology (SOICT 2026)},
  year={2026},
  note={Springer CCIS; not yet submitted}
}
```

---

<p align="center">
  <i>© 2026 Cao, Trinh &amp; Tran · Faculty of Information Technology, HUFLIT · All Rights Reserved</i>
</p>

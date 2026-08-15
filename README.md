<p align="center">
  <img src="assets/soict-hust.png" height="80" alt="HUST SOICT Logo" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/hcmus.png" height="80" alt="HCMUS VNU-HCM Logo" />
</p>

<h1 align="center">SOICT 2026 — LLM-Augmented Log Anomaly Detection & RCA</h1>

<p align="center">
  <a href="https://soict.org"><img src="https://img.shields.io/badge/Conference-SOICT%202026-0066CC.svg" alt="Conference" /></a>
  <a href="https://soict.org"><img src="https://img.shields.io/badge/Organizers-HUST%20%7C%20VNU--HCMUS-CC0000.svg" alt="Organizers" /></a>
  <a href="https://springer.com"><img src="https://img.shields.io/badge/Proceedings-Springer%20CCIS-FF6600.svg" alt="Proceedings" /></a>
  <img src="https://img.shields.io/badge/Indexing-Scopus%20%7C%20EI%20Compendex-6A0DAD.svg" alt="Indexing" />
  <img src="https://img.shields.io/badge/Status-In%20Progress-yellow.svg" alt="Status" />
</p>

---

## 📌 Paper Metadata

| Field | Info |
|---|---|
| **Title** | *LLM-Augmented Log Anomaly Detection: Automated Root Cause Narration for AIOps using Lightweight Foundation Models on Real-World Campus Infrastructure* |
| **Track** | Software Engineering, Trusted Digital Platforms, and Smart Services (Secondary: Applied AI) |
| **Authors** | Trinh Hoang Tu, ThS. Cao Tiến Thành |
| **Affiliation** | Faculty of Information Technology, HUFLIT |
| **Venue** | Ho Chi Minh City, Vietnam — December 4–5, 2026 |
| **Submission Deadline** | Full Paper: September 16, 2026 |

---

## 💡 Overview

Modern enterprise campus networks generate massive volumes of heterogeneous, unstructured log streams. While deep learning models achieve high accuracy in sequence anomaly detection, traditional AIOps platforms suffer from a **critical explainability bottleneck**: raw anomaly scores fail to communicate actionable root causes, forcing SOC analysts to manually triage thousands of log messages.

This paper proposes an **end-to-end LLM-Augmented AIOps Framework** with 3 tightly integrated stages:

```
[Raw Campus Logs 19.28 GB]
        │
        ▼
┌─────────────────────────────┐
│  Stage 1: Drain3 Parsing    │  ← Streaming online log template extraction
└─────────────┬───────────────┘
              │ Template ID sequences
              ▼
┌─────────────────────────────┐
│  Stage 2: TCN-Transformer   │  ← Masked sequence autoencoder anomaly detection
│  Autoencoder                │
└─────────────┬───────────────┘
              │ Anomaly score > threshold
              ▼
┌─────────────────────────────┐
│  Stage 3: LLM RCA Engine    │  ← Phi-3 / Mistral INT4 + RAG generates
│  (Phi-3 / Mistral)          │     Root Cause + Grounded Playbook
└─────────────────────────────┘
```

---

## 📊 Datasets

| Dataset | Source | Size | Scope |
|---|---|---|---|
| **HUFLIT Campus Logs** | Internal Production (19.28 GB raw) | 19.28 GB (14.85M lines) | Real-world enterprise campus syslog |
| **BGL** | [Loghub/Zenodo](https://github.com/logpai/loghub) | 744 MB (4.75M lines) | Supercomputer event logs |
| **HDFS** | [Loghub/Zenodo](https://github.com/logpai/loghub) | 1.58 GB (11.18M lines) | Distributed cloud storage |
| **UNSW-NB15** | [UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | ~100 MB (254K lines) | Network security telemetry |

---

## 🏗️ Repository Structure

```
soict2026-llm-log-rca/
├── assets/                  # Conference logos (HUST SOICT, HCMUS)
├── data/
│   ├── raw/                 # Raw log files (excluded from Git via .gitignore)
│   └── processed/           # Parsed templates & sequence vectors
├── prototype/
│   ├── drain3/              # Online log parser (log_parser.py)
│   ├── pipeline/            # TCN-Transformer autoencoder (tcn_transformer.py)
│   └── llm/                 # LLM RCA narration engine (rca_narration.py)
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

# 3. Run full pipeline
bash run.sh
```

---

## 📄 BibTeX Citation

```bibtex
@article{tu2026soict,
  title={LLM-Augmented Log Anomaly Detection: Automated Root Cause Narration for AIOps
         using Lightweight Foundation Models on Real-World Campus Infrastructure},
  author={Tu, Trinh Hoang and Thanh, Cao Tien},
  journal={Manuscript submitted to the 15th International Symposium on Information
           and Communication Technology (SOICT 2026)},
  year={2026},
  note={Under review, Springer CCIS}
}
```

---

<p align="center">
  <i>© 2026 Trinh Hoang Tu · Faculty of Information Technology, HUFLIT · All Rights Reserved</i>
</p>

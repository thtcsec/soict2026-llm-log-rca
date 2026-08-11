# SOICT 2026 - LLM-Augmented Log Anomaly Detection & RCA

[![Conference](https://img.shields.io/badge/Conference-SOICT%202026-blue.svg)](https://soict.org)
[![Organizers](https://img.shields.io/badge/Organizers-HUST%20%7C%20VNU--HCMUS-red.svg)](https://soict.org)
[![Proceedings](https://img.shields.io/badge/Proceedings-Springer%20CCIS-orange.svg)](https://springer.com)
[![Indexing](https://img.shields.io/badge/Indexing-Scopus%20%7C%20EI%20Compendex-purple.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Research artifact repository for paper submitted to **The 15th International Symposium on Information and Communication Technology (SOICT 2026)**, co-organized by **Hanoi University of Science and Technology (HUST - ĐHBK Hà Nội)** and **University of Science, VNU-HCM (HCMUS)**.

---

## 📌 Paper Metadata

- **Title**: *LLM-Augmented Log Anomaly Detection: Automated Root Cause Narration for AIOps using Lightweight Foundation Models on Real-World Campus Infrastructure*
- **Authors**:
  - **Trịnh Hoàng Tú** (*Department of Cybersecurity, Faculty of Information Technology, HUFLIT*)
  - **ThS. Cao Tiến Thành** (*Department of Cybersecurity, Faculty of Information Technology, HUFLIT*)
- **Track**: *AI Foundations, Foundation Models, and Generative AI* / *Applied AI, Big Data Analytics*
- **Venue & Dates**: Ho Chi Minh City, Vietnam | December 4–5, 2026

---

## 💡 Overview & Architecture

Modern AIOps platforms require not only detecting log anomalies but also providing instantaneous, actionable **Root Cause Analysis (RCA)** for Security Operations Center (SOC) engineers. This paper introduces an end-to-end framework combining:
1. **Drain3 Online Log Parsing**: Converts unstructured syslog streams into structured template IDs.
2. **TCN-Transformer Autoencoder**: Detects complex sequential anomalies at microsecond latency.
3. **Lightweight LLM RCA Narration Engine**: Fine-tuned Phi-3 / Mistral model generating structured natural language incident reports, severity scores, and remediation steps.

Evaluated on **5GB raw production campus network logs** from HUFLIT alongside standard BGL and HDFS benchmarks.

---

## 📁 Repository Structure

```
soict2026/
├── data/                    # Dataset directory (5GB raw campus logs + BGL/HDFS)
├── prototype/
│   ├── drain3/              # Log parsing wrapper
│   ├── pipeline/            # TCN-Transformer model code
│   └── llm/                 # LLM RCA narration generator
├── results/                 # Evaluation CSVs & metric tables
├── paper/                   # Springer CCIS LaTeX source & PDF
│   ├── figures/             # 300 DPI publication graphics
│   └── soict2026.tex        # Main LaTeX manuscript
├── requirements.txt         # Dependencies
├── run.sh                   # 1-Click execution script
└── README.md
```

---

## 🚀 Quickstart Guide

```bash
# Clone repository
git clone https://github.com/thtcsec/soict2026-llm-log-rca.git
cd soict2026-llm-log-rca

# Install dependencies
pip install -r requirements.txt

# Run master evaluation pipeline
bash run.sh
```

---

## 📄 Citation & BibTeX

```bibtex
@inproceedings{tu2026soict,
  title={LLM-Augmented Log Anomaly Detection: Automated Root Cause Narration for AIOps using Lightweight Foundation Models on Real-World Campus Infrastructure},
  author={Tu, Trinh Hoang and Thanh, Cao Tien},
  booktitle={Proceedings of the 15th International Symposium on Information and Communication Technology (SOICT 2026)},
  series={Communications in Computer and Information Science (CCIS)},
  publisher={Springer},
  year={2026},
  address={Ho Chi Minh City, Vietnam}
}
```

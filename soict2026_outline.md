# SOICT 2026 Paper Outline & Technical Specification

**Target Venue**: 15th International Symposium on Information and Communication Technology (SOICT 2026)
**Organizers**: Hanoi University of Science and Technology (HUST - ĐHBK Hà Nội) & VNU-HCMUS
**Proceedings**: Springer CCIS (Communications in Computer and Information Science), Scopus / EI Compendex Indexed
**Target Track**: *Software Engineering, Trusted Digital Platforms, and Smart Services* (backup: Applied AI)
**Submission Deadline**: September 16, 2026 (Abstract: September 9, 2026)
**Conference Date**: December 4–5, 2026 (Ho Chi Minh City, Vietnam)
**Artifact**: https://github.com/thtcsec/soict2026-llm-log-rca (currently **private**)

---

## 📌 Proposed Paper Title

> **"Campus Log Anomaly Detection under Proxy Labels: Validity Limits of Archive Splits and Masked TCN–Transformers"**

---

## 👥 Authors & Affiliations

Order (as on manuscript): **Tú → Thành → Hà**

- **Tu Hoang Trinh∗** (*Faculty of Information Technology, HUFLIT*) — corresponding / submitter: 23dh113972@st.huflit.edu.vn
- **Thanh Tien Cao** (*Faculty of Information Technology, HUFLIT*)
- **Ha Manh Tran** (*PGS.TS; Rector, HUFLIT, 2026–2030*) — research: networks, distributed systems, data mining, computer security, digital transformation

---

## 💡 Core Research Contributions & Novelty

1. **End-to-End AIOps Pipeline**:
   - **Stage 1 (Parsing)**: custom token-similarity parsing for converting unstructured messages into template sequences; it is not Drain3.
   - **Stage 2 (Sequence Anomaly Detection)**: Hybrid TCN-Transformer Deep Autoencoder for detecting microsecond-level anomalous sequence patterns.
   - **Stage 3 (RCA design only)**: proposed local LLM+RAG prompt and JSON schema; the current artifact does not run a model or retrieval system.

2. **Real-World 5GB Enterprise Campus Dataset**:
   - Evaluated on a massive **5GB raw production campus network log dataset** collected from HUFLIT enterprise infrastructure (firewalls, active directory, DNS, authentication, and core routing servers), complemented by standard benchmarks (BGL, HDFS).

3. **Empirical Benchmarks**:
   - Detector metrics: regenerate from `results/tables/huflit_baselines.json` after corrected masked training and leave-one-position-out scoring.
   - Latency: report batch-1 detector wall time separately from batched throughput.
   - LLM RCA evaluation: deferred; no ROUGE/BLEU, expert-ground-truth, or latency claim is permitted for the mock.

---

## 🏗️ Paper Structure (Springer CCIS Template - Max 12 Pages)

- **Section 1: Introduction**: Background on AIOps, limitations of traditional log anomaly detection (lack of explainability), research objectives, and 4 core contributions.
- **Section 2: Related Work**: log parsing, DeepLog/LogAnomaly, Transformer–TCN detectors, and LLM-assisted diagnostics.
- **Section 3: System Architecture & Methodology**:
  - 3.1 Custom Similarity Template Parsing & Feature Vectorization.
  - 3.2 TCN-Transformer Autoencoder Anomaly Detection Architecture.
  - 3.3 Local RCA Narration Design (not implemented or evaluated).
- **Section 4: Experimental Evaluation**:
  - 4.1 Datasets: Real-World 5GB HUFLIT Campus Logs + BGL + HDFS.
  - 4.2 Detection Performance (Precision, Recall, F1 vs Baselines).
  - 4.3 RCA evaluation status and requirements for future expert-grounded testing.
  - 4.4 Inference Latency & Scalability under High Load.
- **Section 5: Case Studies & Qualitative Analysis**: Walkthrough of 3 real incident cases (Brute-Force Auth, DDoS / SYN Flood, Inter-VLAN Routing Loop).
- **Section 6: Conclusion & Future Directions**.

---

## 🛠️ System Implementation Plan

```
soict2026/
├── data/
│   ├── raw/                 # Raw 5GB HUFLIT logs & benchmark datasets
│   └── processed/           # Parsed templates & sequence vectors
├── prototype/
│   ├── drain3/              # Legacy simplified parser; not the Drain3 package
│   ├── pipeline/            # TCN-Transformer PyTorch anomaly detection
│   └── llm/                 # Unevaluated RCA prompt/schema mock
├── results/
│   ├── figures/             # 300 DPI plots for Springer CCIS paper
│   └── tables/              # Evaluation CSVs & metrics
├── paper/
│   ├── figures/             # Embedded graphics
│   ├── llncs.cls            # Springer LNCS / CCIS class file
│   └── soict2026.tex        # Main LaTeX manuscript
└── README.md
```

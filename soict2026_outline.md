# SOICT 2026 Paper Outline & Technical Specification

**Target Venue**: 15th International Symposium on Information and Communication Technology (SOICT 2026)  
**Organizers**: Hanoi University of Science and Technology (HUST - ĐHBK Hà Nội) & VNU-HCMUS  
**Proceedings**: Springer CCIS (Communications in Computer and Information Science), Scopus / EI Compendex Indexed  
**Target Track**: *AI Foundations, Foundation Models, and Generative AI* / *Applied AI, Big Data Analytics*  
**Submission Deadline**: September 16, 2026 (Abstract: September 9, 2026)  
**Conference Date**: December 4–5, 2026 (Ho Chi Minh City, Vietnam)  

---

## 📌 Proposed Paper Title

> **"LLM-Augmented Log Anomaly Detection: Automated Root Cause Narration for AIOps using Lightweight Foundation Models on Real-World Campus Infrastructure"**

---

## 👥 Authors & Affiliations

- **Trịnh Hoàng Tú** (*Department of Cybersecurity, Faculty of Information Technology, HUFLIT*)
- **ThS. Cao Tiến Thành** (*Department of Cybersecurity, Faculty of Information Technology, HUFLIT*)

---

## 💡 Core Research Contributions & Novelty

1. **End-to-End AIOps Pipeline**:
   - **Stage 1 (Online Parsing)**: Drain3 incremental log parsing for stream-processing high-throughput unstructured log messages into structured template sequences.
   - **Stage 2 (Sequence Anomaly Detection)**: Hybrid TCN-Transformer Deep Autoencoder for detecting microsecond-level anomalous sequence patterns.
   - **Stage 3 (Foundation Model RCA Narration)**: Lightweight LLM (Microsoft Phi-3 Mini / Mistral-7B quantized to INT4/INT8 via LoRA / RAG) that ingests anomalous log sequences and generates human-readable Root Cause Analysis (RCA) narratives and actionable remediation playbooks.

2. **Real-World 5GB Enterprise Campus Dataset**:
   - Evaluated on a massive **5GB raw production campus network log dataset** collected from HUFLIT enterprise infrastructure (firewalls, active directory, DNS, authentication, and core routing servers), complemented by standard benchmarks (BGL, HDFS).

3. **Empirical Benchmarks**:
   - F1-Score: $>0.952$ on campus logs.
   - Latency: $< 45\text{ ms}$ for detection + $< 1.2\text{ s}$ for full LLM RCA generation.
   - LLM RCA Evaluation: ROUGE-L / BLEU scores comparing LLM-generated RCA vs expert ground truth.

---

## 🏗️ Paper Structure (Springer CCIS Template - Max 12 Pages)

- **Section 1: Introduction**: Background on AIOps, limitations of traditional log anomaly detection (lack of explainability), research objectives, and 4 core contributions.
- **Section 2: Related Work**: Log parsing (Drain3, Spell), Deep Learning Log Detection (DeepLog, LogAnomaly, TCN-Transformer), LLMs for System Diagnostics.
- **Section 3: System Architecture & Methodology**:
  - 3.1 Streaming Drain3 Log Parsing & Feature Vectorization.
  - 3.2 TCN-Transformer Autoencoder Anomaly Detection Architecture.
  - 3.3 LLM-Based Root Cause Narration Engine (Prompt Engineering, RAG Context Retrieval, LoRA Quantization).
- **Section 4: Experimental Evaluation**:
  - 4.1 Datasets: Real-World 5GB HUFLIT Campus Logs + BGL + HDFS.
  - 4.2 Detection Performance (Precision, Recall, F1 vs Baselines).
  - 4.3 Root Cause Narration Quality (ROUGE-L, Human Expert Triage Time Reduction).
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
│   ├── drain3/              # Drain3 online parser wrapper
│   ├── pipeline/            # TCN-Transformer PyTorch anomaly detection
│   └── llm/                 # Lightweight LLM RCA narration engine
├── results/
│   ├── figures/             # 300 DPI plots for Springer CCIS paper
│   └── tables/              # Evaluation CSVs & metrics
├── paper/
│   ├── figures/             # Embedded graphics
│   ├── llncs.cls            # Springer LNCS / CCIS class file
│   └── soict2026.tex        # Main LaTeX manuscript
└── README.md
```

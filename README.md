<p align="center">
  <img src="assets/soict-hust.png" height="80" alt="HUST SOICT Logo" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/hcmus.png" height="80" alt="HCMUS VNU-HCM Logo" />
  <br />
  <sub>Conference organizers. Authors are affiliated with HUFLIT, not HUST or HCMUS.</sub>
</p>

<h1 align="center">Campus Log Anomaly Detection under Proxy Labels</h1>
<p align="center"><em>Validity limits of archive splits and masked TCN–Transformers</em></p>

<p align="center">
  <a href="https://soict.org"><img src="https://img.shields.io/badge/SOICT-2026-0066CC.svg" alt="SOICT 2026" /></a>
  <a href="https://github.com/thtcsec/soict2026-llm-log-rca/tree/soict2026-camera-ready"><img src="https://img.shields.io/badge/artifact-camera--ready-2ea44f.svg" alt="Camera-ready artifact tag" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/code%20license-MIT-blue.svg" alt="MIT license" /></a>
  <a href="ARTIFACT.md"><img src="https://img.shields.io/badge/reproducibility-tables-informational.svg" alt="Tracked metric tables" /></a>
</p>

## Paper metadata

| Field | Record |
|---|---|
| **Title** | *Campus Log Anomaly Detection under Proxy Labels: Validity Limits of Archive Splits and Masked TCN–Transformers* |
| **Authors** | Hoang Tu Trinh, Thanh Tien Cao, and Ha Manh Tran* |
| **Affiliation** | Faculty of Information Technology, HUFLIT |
| **Venue** | SOICT 2026, Vung Tau / Ho Chi Minh City, 4–5 December 2026 |
| **Track** | Software Engineering, Trusted Digital Platforms, and Smart Services |
| **Status** | Accepted; Springer CCIS forthcoming. No DOI is claimed. |
| **Artifact tag** | [`soict2026-camera-ready`](https://github.com/thtcsec/soict2026-llm-log-rca/tree/soict2026-camera-ready) |
| **Code license** | MIT (`LICENSE`). Campus raw logs are not released. |

\*Corresponding author: `hatm@huflit.edu.vn`

## Overview

Archive packaging and heuristic proxy labels can make log anomaly evaluation look stronger than ranking warrants. On selective HUFLIT backup extracts we compare an archive-order split (Protocol A) with a jointly decontaminated backup-day protocol (Protocol B) under HTTP/keyword proxy labels — **not** SOC ground truth.

The parser is a **custom similarity parser**, not Drain3. We do not claim GraphWalk contributions and do not reuse CSoNet tables. We do **not** claim operational incident detection or a measured LLM/RCA system.

## Headline results

| Setting | F1 | AUC |
|---|---|---|
| Masked TCN–Transformer | 0.0395±0.0163 | 0.7226±0.0178 |
| Masked Transformer without TCN | 0.3475±0.4097 | 0.7685±0.0351 |
| Protocol A always-anomaly baseline | ≈0.892 | — (prevalence 80.5%) |

Switching protocols changes detector rankings. The TCN branch does not justify its leave-one-out scoring cost on these extracts.

## Reproduce

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python prototype/pipeline/run_baselines_huflit.py --threads 2
```

Raw logs stay in `data/raw/` (gitignored). Metric dumps: `results/tables/`. Source: `paper/soict2026.tex`. PDF after SpringerLink.

See `SUBMISSION_INTEGRITY.md` for the CSoNet companion disclosure.

## Limitations

Proxy labels are export heuristics. Protocol B's three decontamination factors are not a causal decomposition. Validation-tuned F1 is diagnostic, not a deployable operating point.

## Citation

```bibtex
@inproceedings{trinh2026proxylabels,
  author    = {Hoang Tu Trinh and Thanh Tien Cao and Ha Manh Tran},
  title     = {Campus Log Anomaly Detection under Proxy Labels: Validity Limits of Archive Splits and Masked TCN--Transformers},
  booktitle = {Proc. 15th Int. Symp. Information and Communication Technology (SOICT)},
  year      = {2026},
  note      = {Accepted manuscript. Artifact: https://github.com/thtcsec/soict2026-llm-log-rca/tree/soict2026-camera-ready}
}
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff). Evaluation protocol: [`ARTIFACT.md`](ARTIFACT.md).

## License

Code is MIT. Do not redistribute HUFLIT campus logs from this repository.

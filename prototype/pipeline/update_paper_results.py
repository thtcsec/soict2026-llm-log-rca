"""
update_paper_results.py
-----------------------
Reads real_multidataset_benchmark.json and cleanly updates soict2026.tex
without string escaping bugs (\t -> tab).

Usage:
    python prototype/pipeline/update_paper_results.py
"""

import json
import re
from pathlib import Path

RESULTS_FILE = Path("results/tables/real_multidataset_benchmark.json")
PAPER_TEX    = Path("paper/soict2026.tex")


def load_results():
    if not RESULTS_FILE.exists():
        print(f"[ERROR] Results file not found: {RESULTS_FILE}")
        return None
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(v, decimals=4):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def build_dataset_table_rows(results):
    sizes = {
        "HUFLIT Campus Logs (Real)":        ("19.28 GB", "14,850,220"),
        "BGL Supercomputer (Blue Gene/L)":  ("744 MB",   "4,747,963"),
        "HDFS Distributed Cluster Logs":    ("1.58 GB",  "11,175,629"),
        "UNSW-NB15 Network Intrusion":      ("100 MB",   "254,004"),
    }
    rows = []
    for r in results:
        name = r["name"]
        size, loglines = sizes.get(name, ("—", "—"))
        if "HUFLIT" in name:
            row = (
                r"\textbf{HUFLIT Campus Logs (Real)} & "
                r"\textbf{" + size + r"} & "
                r"\textbf{" + loglines + r"} & "
                r"\textbf{" + fmt(r['precision']) + r"} & "
                r"\textbf{" + fmt(r['recall']) + r"} & "
                r"\textbf{" + fmt(r['f1_score']) + r"} & "
                r"\textbf{" + fmt(r['roc_auc']) + r"} & "
                r"\textbf{" + fmt(r['latency_ms_per_seq']) + r"} \\"
            )
        else:
            short_name = name.replace(" (Real)", "").replace(" (Blue Gene/L)", "")
            row = (
                f"{short_name} & {size} & {loglines} & "
                f"{fmt(r['precision'])} & {fmt(r['recall'])} & "
                f"{fmt(r['f1_score'])} & {fmt(r['roc_auc'])} & "
                f"{fmt(r['latency_ms_per_seq'])} \\\\"
            )
        rows.append(row)
    return rows


def main():
    results = load_results()
    if not results:
        return

    print("[+] Loaded results:")
    for r in results:
        auc_str = fmt(r['roc_auc'])
        print(f"    {r['name']}: F1={fmt(r['f1_score'])} AUC={auc_str} Lat={fmt(r['latency_ms_per_seq'])}ms")

    huflit = next((r for r in results if "HUFLIT" in r["name"]), None)
    if not huflit:
        print("[WARN] HUFLIT result not found")
        return

    tex = PAPER_TEX.read_text(encoding="utf-8")

    # Re-build dataset table
    rows = build_dataset_table_rows(results)
    new_rows_block = "\n".join(rows)

    pattern = re.compile(
        r"(\\label\{tab_datasets\}.*?\\midrule\n)"
        r"(.*?)"
        r"(\\bottomrule)",
        re.DOTALL
    )
    tex = pattern.sub(r"\g<1>" + new_rows_block + "\n" + r"\g<3>", tex, count=1)

    # Re-build Table 2 (HUFLIT accuracy comparison row)
    acc_pattern = re.compile(
        r"(\\textbf\{TCN-Transformer \(Ours\)\}.*?\\\\)",
        re.DOTALL
    )
    new_acc_row = (
        r"\textbf{TCN-Transformer (Ours)} & "
        r"\textbf{" + fmt(huflit['precision']) + r"} & "
        r"\textbf{" + fmt(huflit['recall']) + r"} & "
        r"\textbf{" + fmt(huflit['f1_score']) + r"} & "
        r"\textbf{" + fmt(huflit['roc_auc']) + r"} \\"
    )
    tex = acc_pattern.sub(new_acc_row, tex, count=1)

    PAPER_TEX.write_text(tex, encoding="utf-8")
    print(f"\n[+] Cleanly patched LaTeX file: {PAPER_TEX.resolve()}")


if __name__ == "__main__":
    main()

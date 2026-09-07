"""
run_multidataset_benchmarks.py — NO hardcoded paper numbers.
Retired compatibility entrypoint; public-dataset runs remain deferred.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "multidataset_benchmark.json"
HUFLIT = ROOT / "results" / "tables" / "huflit_hardened_results.json"


def main():
    if not HUFLIT.exists():
        raise SystemExit(f"Missing {HUFLIT}; run harden_huflit_train.py first")
    h = json.loads(HUFLIT.read_text(encoding="utf-8"))
    head = h["headline"]
    results = [
        {
            "name": "HUFLIT Campus Logs (hardened stream)",
            "source": str(HUFLIT),
            "lines": h["ingest_meta"]["lines"],
            "prec": round(head["precision"], 4),
            "rec": round(head["recall"], 4),
            "f1": round(head["f1"], 4),
            "auc": round(head["roc_auc"], 4),
            "auprc": round(head["auprc"], 4),
            "latency_ms": round(head["latency_ms_per_seq"], 4),
            "status": "measured",
        },
        {"name": "BGL", "status": "deferred"},
        {"name": "HDFS", "status": "deferred"},
        {"name": "UNSW-NB15", "status": "deferred"},
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[+] wrote {OUT}")


if __name__ == "__main__":
    raise SystemExit(
        "Retired: huflit_hardened_results.json used the pre-fix masking path. "
        "Use huflit_baselines.json; BGL/HDFS/UNSW results remain deferred."
    )

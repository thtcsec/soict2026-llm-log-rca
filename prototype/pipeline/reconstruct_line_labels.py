"""
reconstruct_line_labels.py
==========================
Bounded streaming reconstruction of Protocol-B *window line-label vectors*
using the same ZIP/member order, SHA-1 dedup, caps, and proxy rules as
build_huflit_v2_split.py — WITHOUT re-parsing templates and WITHOUT storing
raw log text.

Validates that OR aggregation of reconstructed line labels equals the stored
window OR labels exactly. On success, enables scientifically valid k-of-L
sensitivity (Phase 3).

Outputs (gitignored / private intermediates):
  data/private/huflit_v2_line_labels/{train,val,test}_line_labels.npy  uint8 (N,20)
  data/private/huflit_v2_line_labels/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse exact semantics from the original builder
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype"))
from pipeline.build_huflit_v2_split import (  # noqa: E402
    DAY_TEST,
    DAY_TRAIN,
    DAY_VAL,
    discover_members,
    is_anomaly_line,
    source_from_zip,
)

DATA = ROOT / "data" / "processed" / "huflit_v2"
OUT = ROOT / "data" / "private" / "huflit_v2_line_labels"
META = ROOT / "results" / "tables" / "huflit_v2_split_meta.json"


def stream_label_records(members, max_lines: int, per_day_cap: int, per_file_cap: int):
    """Same keep/skip logic as stream_deduped_records, but store labels only."""
    seen: set[bytes] = set()
    day_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    kept = 0
    skipped_dup = 0
    raw_seen = 0
    # day, src, member, label
    records: list[tuple[str, str, str, int]] = []

    for day, zpath, member, size in members:
        if kept >= max_lines:
            break
        if day_counts[day] >= per_day_cap:
            continue
        src = source_from_zip(zpath)
        local = 0
        print(
            f"  • {day}/{zpath.name} :: {Path(member).name} "
            f"({size/1e6:.1f} MB) [{src}]",
            flush=True,
        )
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                with zf.open(member, "r") as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                    for line in text:
                        line = line.strip()
                        if not line:
                            continue
                        raw_seen += 1
                        h = hashlib.sha1(line.encode("utf-8", errors="replace")).digest()
                        if h in seen:
                            skipped_dup += 1
                            continue
                        if (
                            day_counts[day] >= per_day_cap
                            or kept >= max_lines
                            or local >= per_file_cap
                        ):
                            break
                        seen.add(h)
                        lab = int(is_anomaly_line(line))
                        records.append((day, src, member, lab))
                        day_counts[day] += 1
                        source_counts[src] += 1
                        kept += 1
                        local += 1
        except Exception as e:
            print(f"    [!] {e}", flush=True)

    meta = {
        "raw_lines_scanned_nonempty": raw_seen,
        "unique_kept": kept,
        "duplicates_skipped": skipped_dup,
        "dup_rate_among_scanned": skipped_dup / max(raw_seen, 1),
        "lines_per_day": dict(day_counts),
        "lines_per_source": dict(source_counts),
    }
    return records, meta


def build_line_label_windows(records, window: int, step: int):
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for day, src, member, lab in records:
        groups[(day, src, member)].append(lab)

    by_split = {"train": [], "val": [], "test": []}
    group_stats = []
    for (day, src, member), labs in groups.items():
        if day in DAY_TRAIN:
            split = "train"
        elif day in DAY_VAL:
            split = "val"
        elif day in DAY_TEST:
            split = "test"
        else:
            continue
        n_win = 0
        if len(labs) >= window:
            for i in range(0, len(labs) - window + 1, step):
                by_split[split].append(labs[i : i + window])
                n_win += 1
        group_stats.append(
            {
                "day": day,
                "source": src,
                "member": member,
                "split": split,
                "lines": len(labs),
                "windows": n_win,
                "line_anomaly_rate": float(np.mean(labs)) if labs else 0.0,
            }
        )

    out = {}
    for name, rows in by_split.items():
        if rows:
            out[name] = np.asarray(rows, dtype=np.uint8)
        else:
            out[name] = np.zeros((0, window), dtype=np.uint8)
    return out, group_stats


def always_anomaly_f1(y: np.ndarray) -> float:
    p = float(y.mean()) if len(y) else 0.0
    return float(2 * p / (1 + p)) if p > 0 else 0.0


def preflight(data_root: Path, max_member_mb: float, seconds: float = 20.0):
    members = discover_members(data_root, max_member_mb)
    print(f"Preflight: {len(members)} eligible members", flush=True)
    if not members:
        raise SystemExit("No members found")
    # time scanning first member only
    day, zpath, member, size = members[0]
    t0 = time.perf_counter()
    n = 0
    with zipfile.ZipFile(zpath, "r") as zf:
        with zf.open(member, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            for line in text:
                if line.strip():
                    n += 1
                if time.perf_counter() - t0 >= seconds:
                    break
    elapsed = time.perf_counter() - t0
    rate = n / max(elapsed, 1e-6)
    # rough: original scanned ~1.26M nonempty lines
    est_full = 1_262_211 / max(rate, 1.0)
    print(
        f"Preflight rate ≈ {rate:.0f} nonempty lines/s on {zpath.name}::{Path(member).name}; "
        f"est. full scan ≈ {est_full/60:.1f} min (label-only, no parser)",
        flush=True,
    )
    return members, est_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"D:\huflit-campus-logs")
    ap.add_argument("--max_lines", type=int, default=1_000_000)
    ap.add_argument("--per_day_cap", type=int, default=200_000)
    ap.add_argument("--per_file_cap", type=int, default=50_000)
    ap.add_argument("--max_member_mb", type=float, default=500.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--skip_preflight", action="store_true")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not META.exists():
        raise SystemExit(f"Missing original meta: {META}")
    orig_meta = json.loads(META.read_text(encoding="utf-8"))

    if not args.skip_preflight:
        members, est = preflight(data_root, args.max_member_mb)
        if args.preflight_only:
            return
        if est > 6 * 3600:
            raise SystemExit(
                f"Estimated reconstruction {est/3600:.1f}h exceeds 6h budget; aborting."
            )
    else:
        members = discover_members(data_root, args.max_member_mb)

    print("=" * 72, flush=True)
    print(" Reconstruct Protocol-B window line labels (label-only stream)", flush=True)
    print("=" * 72, flush=True)
    t0 = time.perf_counter()
    records, stream_meta = stream_label_records(
        members, args.max_lines, args.per_day_cap, args.per_file_cap
    )
    print(
        f"Kept {stream_meta['unique_kept']:,} | dups skipped {stream_meta['duplicates_skipped']:,} "
        f"in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    parts, group_stats = build_line_label_windows(records, args.window, args.step)

    # --- Mandatory validation ---
    stored = {
        "train": np.load(DATA / "train_labels.npy").astype(np.int64),
        "val": np.load(DATA / "val_labels.npy").astype(np.int64),
        "test": np.load(DATA / "test_labels.npy").astype(np.int64),
    }
    validation = {}
    ok = True
    for split in ("train", "val", "test"):
        ll = parts[split]
        y_or = (ll.sum(axis=1) >= 1).astype(np.int64) if len(ll) else np.zeros(0, dtype=np.int64)
        y_st = stored[split]
        shape_ok = ll.shape == (len(y_st), args.window) if len(y_st) else ll.shape[0] == 0
        equal = bool(np.array_equal(y_or, y_st)) if shape_ok and len(y_st) == len(y_or) else False
        validation[split] = {
            "line_labels_shape": list(ll.shape),
            "stored_or_n": int(len(y_st)),
            "recon_or_n": int(len(y_or)),
            "shape_ok": bool(shape_ok),
            "or_exact_match": equal,
            "recon_prevalence": float(y_or.mean()) if len(y_or) else 0.0,
            "stored_prevalence": float(y_st.mean()) if len(y_st) else 0.0,
            "recon_always_anomaly_f1": always_anomaly_f1(y_or),
            "mismatch_count": int(np.sum(y_or != y_st)) if equal is False and len(y_or) == len(y_st) else 0,
        }
        print(f"  {split}: {validation[split]}", flush=True)
        if not equal:
            ok = False

    # Compare stream meta against original
    om = orig_meta.get("stream_meta", {})
    meta_cmp = {
        "unique_kept_match": stream_meta["unique_kept"] == om.get("unique_kept"),
        "duplicates_skipped_match": stream_meta["duplicates_skipped"] == om.get("duplicates_skipped"),
        "raw_scanned_match": stream_meta["raw_lines_scanned_nonempty"]
        == om.get("raw_lines_scanned_nonempty"),
        "lines_per_day_match": stream_meta["lines_per_day"] == om.get("lines_per_day"),
        "reconstructed": stream_meta,
        "original": om,
    }
    print("Stream meta comparison:", json.dumps(meta_cmp, indent=2)[:1200], flush=True)

    if not ok:
        print("STOP: OR reconstruction did not match 100%. Not writing aligned artifacts.", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        fail_path = OUT / "FAILED_manifest.json"
        fail_path.write_text(
            json.dumps(
                {
                    "status": "FAILED_OR_MISMATCH",
                    "validation": validation,
                    "stream_meta_comparison": meta_cmp,
                    "elapsed_sec": time.perf_counter() - t0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise SystemExit(2)

    OUT.mkdir(parents=True, exist_ok=True)
    for split, arr in parts.items():
        np.save(OUT / f"{split}_line_labels.npy", arr)

    # Note: template-window equality not claimed — label-only stream; different
    # raw lines may share templates. Alignment is validated via OR label identity
    # and window counts / group line counts vs group_stats.json when available.
    gs_path = DATA / "group_stats.json"
    group_cmp = None
    if gs_path.exists():
        orig_gs = json.loads(gs_path.read_text(encoding="utf-8"))
        key = lambda g: (g["day"], g["source"], g["member"])
        omap = {key(g): g for g in orig_gs}
        diffs = []
        for g in group_stats:
            o = omap.get(key(g))
            if o is None or o["lines"] != g["lines"] or o["windows"] != g["windows"]:
                diffs.append({"recon": g, "orig": o})
        group_cmp = {"n_group_diffs": len(diffs), "sample_diffs": diffs[:5]}

    manifest = {
        "status": "OK",
        "purpose": "Protocol-B window line-label reconstruction for k-of-L sensitivity",
        "params": {
            "data_root": str(data_root),
            "max_lines": args.max_lines,
            "per_day_cap": args.per_day_cap,
            "per_file_cap": args.per_file_cap,
            "max_member_mb": args.max_member_mb,
            "window": args.window,
            "step": args.step,
            "proxy_rule": "build_huflit_v2_split.is_anomaly_line (HTTP 4xx/5xx, deny/block, keywords)",
            "parser": "NOT re-run; label-only stream",
            "note_alignment": (
                "OR window labels validated against stored *.npy. Template-ID sequences "
                "were not regenerated; template-window equality is not claimed."
            ),
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "stream_meta": stream_meta,
        "stream_meta_comparison": meta_cmp,
        "validation": validation,
        "group_stats_comparison": group_cmp,
        "elapsed_sec": time.perf_counter() - t0,
        "outputs": {
            split: str(OUT / f"{split}_line_labels.npy") for split in parts
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Non-sensitive summary for results/
    summary = {
        "status": manifest["status"],
        "validation": validation,
        "stream_meta_comparison": {
            k: meta_cmp[k]
            for k in (
                "unique_kept_match",
                "duplicates_skipped_match",
                "raw_scanned_match",
                "lines_per_day_match",
            )
        },
        "elapsed_sec": manifest["elapsed_sec"],
    }
    (ROOT / "results" / "tables" / "line_label_reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"OK — wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

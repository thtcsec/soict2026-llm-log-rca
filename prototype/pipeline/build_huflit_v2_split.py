"""
build_huflit_v2_split.py — Validity-hardened HUFLIT split (SOICT Full Paper)
============================================================================
Does NOT expand FortiGate CSV or members > max_member_mb.

Fixes vs v1 leaky archive split:
  1) Global raw-line hash deduplication (keep first occurrence)
  2) Windows never cross (day, source, member) boundaries
  3) Day-based train/val/test (event-period proxy; not archive position)
  4) Persist window-label definition: OR over L proxy line labels

Outputs under data/processed/huflit_v2/ and results/tables/huflit_v2_split_meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ANOM_KW = re.compile(
    r"\b(error|fail|denied|critical|refused|invalid|attack|flood|drop|reject|"
    r"alert|blocked|unauthorized|forbidden|malware|virus|intrusion|bruteforce|"
    r"action=\"deny\"|action=deny|status=deny)\b",
    re.I,
)
HTTP_STATUS = re.compile(r"\s(\d{3})\s")
FORTI_DENY = re.compile(r'action[=:]"?deny"?|utmaction[=:]"?block"?|attack[=:]"?', re.I)

DAY_TRAIN = ("15062026", "16062026", "17062026")
DAY_VAL = ("18062026",)
DAY_TEST = ("19062026",)


def source_from_zip(zpath: Path) -> str:
    name = zpath.name.lower()
    for key in ("forti", "careerhub", "courses", "portal", "thuvien", "aca", "www"):
        if name.startswith(key) or f"_{key}_" in name or key in name.split("_")[0]:
            if key == "forti":
                return "fortigate"
            return key
    return "other"


def is_anomaly_line(line: str) -> bool:
    m = HTTP_STATUS.search(line)
    if m and 400 <= int(m.group(1)) <= 599:
        return True
    if FORTI_DENY.search(line):
        return True
    return bool(ANOM_KW.search(line))


class SimilarityTemplateParser:
    MASK_TOKENS = [
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
        (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "<UUID>"),
        (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<HEX>"),
        (re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b"), "<TS>"),
        (re.compile(r"\b\d+\.\d+\b"), "<FLOAT>"),
        (re.compile(r"\b\d{5,}\b"), "<NUM>"),
        (re.compile(r"/(?:home|var|usr|etc|tmp|proc|opt)/\S+"), "<PATH>"),
    ]

    def __init__(self, sim_th: float = 0.5, max_vocab: int = 2500):
        self.sim_th = sim_th
        self.max_vocab = max_vocab
        self.templates: list[list[str]] = []
        self.template_to_id: dict[str, int] = {}

    def _sanitize(self, msg: str) -> str:
        for pat, rep in self.MASK_TOKENS:
            msg = pat.sub(rep, msg)
        return msg.strip()

    def _similarity(self, a: list[str], b: list[str]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        return sum(x == y for x, y in zip(a, b)) / len(a)

    def parse(self, raw_line: str) -> int:
        tokens = self._sanitize(raw_line).split()
        if not tokens:
            return 0
        best_idx, best_sim = -1, -1.0
        for i, tmpl in enumerate(self.templates):
            sim = self._similarity(tokens, tmpl)
            if sim > best_sim and sim >= self.sim_th:
                best_sim, best_idx = sim, i
        if best_idx >= 0:
            merged = [a if a == b else "<*>" for a, b in zip(tokens, self.templates[best_idx])]
            self.templates[best_idx] = merged
            tmpl_str = " ".join(merged)
        else:
            if len(self.templates) >= self.max_vocab:
                tmpl_str = " ".join(self.templates[0])
            else:
                self.templates.append(tokens)
                tmpl_str = " ".join(tokens)
        if tmpl_str not in self.template_to_id:
            self.template_to_id[tmpl_str] = len(self.template_to_id) + 1
        return self.template_to_id[tmpl_str]


def discover_members(root: Path, max_member_mb: float):
    items = []
    for zpath in sorted(root.rglob("*.zip")):
        day = zpath.parent.name
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                for info in zf.infolist():
                    name = info.filename.replace("\\", "/")
                    if info.is_dir() or not name.lower().endswith(".log"):
                        continue
                    if "fortigate.csv" in name.lower():
                        continue
                    size_mb = info.file_size / (1024 * 1024)
                    if size_mb > max_member_mb:
                        print(f"  [skip large] {zpath.name}::{name} ({size_mb:.0f} MB)")
                        continue
                    items.append((day, zpath, name, info.file_size))
        except zipfile.BadZipFile as e:
            print(f"  [bad zip] {zpath}: {e}")
    items.sort(key=lambda t: (t[0], t[1].name, t[2]))
    return items


def stream_deduped_records(members, max_lines: int, per_day_cap: int, per_file_cap: int):
    """Yield (day, source, member, tid_placeholder_line, label) after hash dedup."""
    seen: set[bytes] = set()
    day_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    kept = 0
    skipped_dup = 0
    raw_seen = 0
    records: list[tuple[str, str, str, str, int]] = []  # day,src,member,line,label

    for day, zpath, member, size in members:
        if kept >= max_lines:
            break
        if day_counts[day] >= per_day_cap:
            continue
        src = source_from_zip(zpath)
        local = 0
        print(f"  • {day}/{zpath.name} :: {Path(member).name} ({size/1e6:.1f} MB) [{src}]")
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
                        if day_counts[day] >= per_day_cap or kept >= max_lines or local >= per_file_cap:
                            break
                        seen.add(h)
                        lab = int(is_anomaly_line(line))
                        records.append((day, src, member, line, lab))
                        day_counts[day] += 1
                        source_counts[src] += 1
                        kept += 1
                        local += 1
        except Exception as e:
            print(f"    [!] {e}")
        time.sleep(0.02)

    meta = {
        "raw_lines_scanned_nonempty": raw_seen,
        "unique_kept": kept,
        "duplicates_skipped": skipped_dup,
        "dup_rate_among_scanned": skipped_dup / max(raw_seen, 1),
        "lines_per_day": dict(day_counts),
        "lines_per_source": dict(source_counts),
    }
    return records, meta


def build_windows(records, parser: SimilarityTemplateParser, window: int, step: int):
    """Group by (day, source, member); never cross boundaries."""
    groups: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for day, src, member, line, lab in records:
        tid = parser.parse(line)
        groups[(day, src, member)].append((tid, lab))

    by_split = {"train": ([], []), "val": ([], []), "test": ([], [])}
    group_stats = []
    for (day, src, member), seq in groups.items():
        if day in DAY_TRAIN:
            split = "train"
        elif day in DAY_VAL:
            split = "val"
        elif day in DAY_TEST:
            split = "test"
        else:
            continue
        toks = [t for t, _ in seq]
        labs = [y for _, y in seq]
        n_win = 0
        anom_win = 0
        if len(toks) >= window:
            for i in range(0, len(toks) - window + 1, step):
                w_tok = toks[i : i + window]
                w_lab = int(any(labs[i : i + window]))
                by_split[split][0].append(w_tok)
                by_split[split][1].append(w_lab)
                n_win += 1
                anom_win += w_lab
        group_stats.append(
            {
                "day": day,
                "source": src,
                "member": member,
                "split": split,
                "lines": len(toks),
                "windows": n_win,
                "window_anomaly_rate": anom_win / n_win if n_win else 0.0,
                "line_anomaly_rate": float(np.mean(labs)) if labs else 0.0,
            }
        )

    out = {}
    for name, (xs, ys) in by_split.items():
        if xs:
            out[name] = (
                np.asarray(xs, dtype=np.int32),
                np.asarray(ys, dtype=np.int8),
            )
        else:
            out[name] = (np.zeros((0, window), dtype=np.int32), np.zeros((0,), dtype=np.int8))
    return out, group_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=r"D:\huflit-campus-logs")
    ap.add_argument("--max_lines", type=int, default=1_000_000)
    ap.add_argument("--per_day_cap", type=int, default=200_000)
    ap.add_argument("--per_file_cap", type=int, default=50_000)
    ap.add_argument("--max_member_mb", type=float, default=500.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--max_vocab", type=int, default=2500)
    args = ap.parse_args()

    out_dir = ROOT / "data" / "processed" / "huflit_v2"
    res_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" HUFLIT v2 — dedup + per-source windows + day split")
    print(f" Data: {args.data_root}")
    print(f" Train days {DAY_TRAIN} | Val {DAY_VAL} | Test {DAY_TEST}")
    print("=" * 72)

    members = discover_members(Path(args.data_root), args.max_member_mb)
    print(f"Eligible .log members: {len(members)}")
    t0 = time.time()
    records, stream_meta = stream_deduped_records(
        members, args.max_lines, args.per_day_cap, args.per_file_cap
    )
    print(f"Deduped lines kept: {stream_meta['unique_kept']:,} "
          f"(skipped dups {stream_meta['duplicates_skipped']:,}) in {time.time()-t0:.1f}s")

    parser = SimilarityTemplateParser(max_vocab=args.max_vocab)
    parts, group_stats = build_windows(records, parser, args.window, args.step)

    for k, (X, y) in parts.items():
        print(f"  {k:5s}: {len(X):,} windows | anomaly {100*y.mean() if len(y) else 0:.2f}%")

    # Cross-split hash leakage check on windows (template id tuples)
    def win_hashes(X):
        return {hashlib.sha1(row.tobytes()).hexdigest() for row in X}

    h_tr, h_va, h_te = win_hashes(parts["train"][0]), win_hashes(parts["val"][0]), win_hashes(parts["test"][0])
    leak = {
        "train_val_shared_windows": len(h_tr & h_va),
        "train_test_shared_windows": len(h_tr & h_te),
        "val_test_shared_windows": len(h_va & h_te),
        "train_unique_windows": len(h_tr),
        "val_unique_windows": len(h_va),
        "test_unique_windows": len(h_te),
    }
    print("Window hash overlap:", leak)

    np.save(out_dir / "train_sequences.npy", parts["train"][0])
    np.save(out_dir / "train_labels.npy", parts["train"][1])
    np.save(out_dir / "val_sequences.npy", parts["val"][0])
    np.save(out_dir / "val_labels.npy", parts["val"][1])
    np.save(out_dir / "test_sequences.npy", parts["test"][0])
    np.save(out_dir / "test_labels.npy", parts["test"][1])
    with open(out_dir / "templates.json", "w", encoding="utf-8") as f:
        json.dump(parser.template_to_id, f, ensure_ascii=False)

    line_labs = [r[4] for r in records]
    payload = {
        "experiment": "huflit_v2_dedup_source_bounded_day_split",
        "window_label_definition": "window positive iff any of L proxy line labels is positive (OR aggregation)",
        "split": {
            "train_days": list(DAY_TRAIN),
            "val_days": list(DAY_VAL),
            "test_days": list(DAY_TEST),
            "note": "Day folders are backup collection periods, not verified per-event timestamps.",
        },
        "window": args.window,
        "step": args.step,
        "stream_meta": stream_meta,
        "line_anomaly_rate_kept": float(np.mean(line_labs)) if line_labs else 0.0,
        "partition_sizes": {
            k: {
                "n": int(len(v[0])),
                "anomaly_rate": float(v[1].mean()) if len(v[1]) else 0.0,
                "implied_always_anomaly_f1": float(
                    2 * p / (1 + p)
                )
                if (p := float(v[1].mean()) if len(v[1]) else 0.0) > 0
                else 0.0,
            }
            for k, v in parts.items()
        },
        "window_hash_overlap": leak,
        "group_stats_summary": {
            "n_groups": len(group_stats),
            "by_source_windows": {},
        },
        "vocab_size": len(parser.template_to_id) + 2,
        "fortigate_policy": "skip fortigate.csv and members > max_member_mb; no full RAR expand",
    }
    # aggregate windows by source
    src_win = defaultdict(int)
    for g in group_stats:
        src_win[g["source"]] += g["windows"]
    payload["group_stats_summary"]["by_source_windows"] = dict(src_win)

    with open(res_dir / "huflit_v2_split_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "group_stats.json", "w", encoding="utf-8") as f:
        json.dump(group_stats, f)

    print(f"Wrote {out_dir}")
    print(f"Meta → {res_dir / 'huflit_v2_split_meta.json'}")


if __name__ == "__main__":
    main()

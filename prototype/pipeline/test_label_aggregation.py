"""Unit tests for k-of-L aggregation invariants (SOICT camera-ready)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype"))
from pipeline.compute_k_of_l_table import aggregate, always_anomaly_f1  # noqa: E402

DATA = ROOT / "data" / "processed" / "huflit_v2"
LLDIR = ROOT / "data" / "private" / "huflit_v2_line_labels"


def test_synthetic_monotonic():
    ll = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    y1, y2, y3 = aggregate(ll, 1), aggregate(ll, 2), aggregate(ll, 3)
    assert np.array_equal(y1, np.array([0, 1, 1, 1]))
    assert np.array_equal(y2, np.array([0, 0, 1, 1]))
    assert np.array_equal(y3, np.array([0, 0, 0, 1]))
    assert np.all(y3 <= y2)
    assert np.all(y2 <= y1)
    assert always_anomaly_f1(y1) >= always_anomaly_f1(y3)


def test_reconstructed_or_equals_stored():
    if not (LLDIR / "test_line_labels.npy").exists():
        print("SKIP reconstructed OR test (no private line labels)")
        return
    ll = np.load(LLDIR / "test_line_labels.npy")
    y_or = np.load(DATA / "test_labels.npy").astype(np.int64)
    assert ll.shape == (len(y_or), 20)
    assert np.array_equal(aggregate(ll, 1), y_or)
    assert np.all(aggregate(ll, 3) <= aggregate(ll, 2))
    assert np.all(aggregate(ll, 2) <= aggregate(ll, 1))
    p1 = float(aggregate(ll, 1).mean())
    p2 = float(aggregate(ll, 2).mean())
    p3 = float(aggregate(ll, 3).mean())
    assert p1 >= p2 >= p3
    assert always_anomaly_f1(aggregate(ll, 1)) >= always_anomaly_f1(aggregate(ll, 3))


def main():
    test_synthetic_monotonic()
    test_reconstructed_or_equals_stored()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

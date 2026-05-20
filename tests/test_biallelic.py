"""Tests for the biallelic-peak detection pipeline:
  find_biallelic_boundaries -> scan_for_biallelic_signal -> make_combined_boundaries_dict
"""
from collections import Counter

import pandas as pd
import pytest

import support_functions as sf


# ---------------------------------------------------------------------------
# find_biallelic_boundaries
# ---------------------------------------------------------------------------

def _peak_df(peak_to_count):
    """Helper: build a peak_df shaped like `df.groupby('hist_peak').count()`."""
    return pd.DataFrame(
        {"read_num": list(peak_to_count.values()), "other": list(peak_to_count.values())},
        index=pd.Index(list(peak_to_count.keys()), name="hist_peak"),
    )


def test_find_biallelic_boundaries_two_clear_peaks():
    peak_df = _peak_df({10: 100, 15: 5, 20: 95})
    peaks, boundaries, min_peak, diff = sf.find_biallelic_boundaries(peak_df)
    assert list(peaks) == [10, 20]
    assert boundaries[10] == 0
    assert boundaries[20] == 1
    # Mid peak 15 falls into the lower-half bucket (bottom_a contains it).
    assert boundaries[15] == 0
    assert min_peak == 10
    assert diff == 10


def test_find_biallelic_boundaries_dominant_peak_rejected():
    # One peak has >60% of the reads of the top-2 — not biallelic.
    peak_df = _peak_df({10: 1000, 20: 50})
    out = sf.find_biallelic_boundaries(peak_df)
    assert out == (None, None, None, None)


def test_find_biallelic_boundaries_single_peak_rejected():
    peak_df = _peak_df({10: 100})
    out = sf.find_biallelic_boundaries(peak_df)
    assert out == (None, None, None, None)


def test_find_biallelic_boundaries_adjacent_peaks():
    # Adjacent peaks (diff=1). Boundaries still assigned.
    peak_df = _peak_df({10: 100, 11: 95})
    peaks, boundaries, min_peak, diff = sf.find_biallelic_boundaries(peak_df)
    assert list(peaks) == [10, 11]
    assert min_peak == 10
    assert diff == 1
    # 10 ends up in the bottom half, 11 in the top half (or both share a
    # bucket given the rounding) — but the two original peaks should still be
    # mapped to *different* assignments.
    assert boundaries[10] != boundaries[11]


# ---------------------------------------------------------------------------
# scan_for_biallelic_signal
# ---------------------------------------------------------------------------

def _build_test_peaks(cell_assignments):
    """Build a histogram-style DataFrame from a dict of:
       {(cell, t_key): hist_peak}.
    Index: (cell, ins_key, a_key, t_key, tag_site)."""
    rows = []
    for (cell, t_key), hist_peak in cell_assignments.items():
        rows.append(
            (cell, "ins1", 0, t_key, 1000 + hash((cell, t_key)) % 10_000, hist_peak, 50)
        )
    df = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "hist_peak", "read_num"],
    )
    return df.set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])


def test_scan_for_biallelic_signal_single_tract():
    # 100 cells at peak=10, 95 at peak=20, all in t_key=0.
    assignments = {}
    for i in range(100):
        assignments[(f"cell{i:03d}", 0)] = 10
    for i in range(100, 195):
        assignments[(f"cell{i:03d}", 0)] = 20
    df = _build_test_peaks(assignments)
    boundaries_dict, assignment_combinations, t_key_to_order = (
        sf.scan_for_biallelic_signal(df)
    )
    assert 0 in boundaries_dict
    assert sorted(boundaries_dict[0]["biallelic_peaks"]) == [10, 20]
    # Single tract -> no combined assignment dict.
    assert assignment_combinations is None
    assert t_key_to_order is None


def test_scan_for_biallelic_signal_no_signal():
    # One dominant peak -> no biallelic signal anywhere.
    assignments = {f"cell{i:03d}": 10 for i in range(50)}
    df = _build_test_peaks({(c, 0): p for c, p in assignments.items()})
    boundaries_dict, ac, tk = sf.scan_for_biallelic_signal(df)
    assert boundaries_dict == {}
    assert ac is None
    assert tk is None


def test_scan_for_biallelic_signal_multi_tract_builds_combined():
    # Two tracts, both biallelic, with consistent per-cell assignments.
    rows = []
    for i in range(100):
        cell = f"cell{i:03d}"
        # tract 0: low peak; tract 1: low peak
        rows.append((cell, "ins1", 0, 0, 100 + i, 10, 50))
        rows.append((cell, "ins1", 0, 1, 100 + i, 12, 50))
    for i in range(100, 200):
        cell = f"cell{i:03d}"
        # tract 0: high peak; tract 1: high peak
        rows.append((cell, "ins1", 0, 0, 100 + i, 25, 50))
        rows.append((cell, "ins1", 0, 1, 100 + i, 28, 50))
    df = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "hist_peak", "read_num"],
    ).set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])

    boundaries_dict, assignment_combinations, t_key_to_order = (
        sf.scan_for_biallelic_signal(df)
    )
    assert set(boundaries_dict.keys()) == {0, 1}
    # Combined assignments should be tracked when there's more than one tract.
    assert assignment_combinations is not None
    assert set(t_key_to_order.keys()) == {0, 1}
    # All cells either assigned (0,0) or (1,1) — both tracts agree.
    assert sum(assignment_combinations.values()) == 200
    # (0,0) and (1,1) should dominate; (0,1) and (1,0) should be small or absent.
    assert assignment_combinations[(0, 0)] + assignment_combinations[(1, 1)] >= 200
    assert assignment_combinations.get((0, 1), 0) == 0
    assert assignment_combinations.get((1, 0), 0) == 0


# ---------------------------------------------------------------------------
# make_combined_boundaries_dict directly
# ---------------------------------------------------------------------------

def test_make_combined_boundaries_dict_skips_cells_missing_tracts():
    # Cell A has both t_keys; cell B only has t_key=0. Cell B should be skipped.
    rows = [
        ("A", "ins1", 0, 0, 100, 10, 50),
        ("A", "ins1", 0, 1, 100, 25, 50),
        ("B", "ins1", 0, 0, 200, 10, 50),
        # B has no row for t_key=1 -> incomplete -> skipped
    ]
    df = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "hist_peak", "read_num"],
    ).set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])

    boundaries_dict = {
        0: {"biallelic_boundaries": {10: 0, 25: 1}, "biallelic_peaks": [10, 25],
            "min_peak": 10, "difference": 15},
        1: {"biallelic_boundaries": {25: 1, 10: 0}, "biallelic_peaks": [10, 25],
            "min_peak": 10, "difference": 15},
    }
    combinations, t_key_to_order = sf.make_combined_boundaries_dict(boundaries_dict, df)
    # Only cell A contributes.
    assert sum(combinations.values()) == 1
    assert combinations[(0, 1)] == 1

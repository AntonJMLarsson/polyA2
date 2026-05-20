"""Tests for tract/allele phasing helpers used inside get_final_stats_to_write."""
from collections import Counter

import pandas as pd

import support_functions as sf


# ---------------------------------------------------------------------------
# get_allele_index
# ---------------------------------------------------------------------------

def test_get_allele_index_empty_categorization_returns_a_key():
    assert sf.get_allele_index(7, 100, {}) == 7


def test_get_allele_index_tag_site_in_categorization():
    assert sf.get_allele_index(7, 100, {100: 1, 200: 0}) == 1


def test_get_allele_index_tag_site_missing_returns_none():
    assert sf.get_allele_index(7, 100, {200: 0}) is None


# ---------------------------------------------------------------------------
# find_original_alleles
# ---------------------------------------------------------------------------

def test_find_original_alleles_picks_majority_kd():
    # 3 cells, all allele 0, tract 0.
    # Cell A: tag_site 100 -> (kd=5, hist_peak=10, read_num=20)
    # Cell B: tag_site 200 -> (kd=5, hist_peak=10, read_num=15)
    # Cell C: tag_site 300 -> (kd=7, hist_peak=12, read_num=30)
    tract_stats = {
        "A": {0: {0: {100: (5, 10, 20)}}},
        "B": {0: {0: {200: (5, 10, 15)}}},
        "C": {0: {0: {300: (7, 12, 30)}}},
    }
    originals, hist_peaks = sf.find_original_alleles(tract_stats)
    # kd=5 appeared in 2 cells, kd=7 in 1 cell -> 5 wins.
    assert originals[(0, 0)] == 5
    assert hist_peaks[(0, 0)] == 10


def test_find_original_alleles_picks_most_common_hist_peak_for_winner():
    # All same kd, different hist_peaks. The most common hist_peak should win.
    tract_stats = {
        "A": {0: {0: {100: (5, 10, 10)}}},
        "B": {0: {0: {200: (5, 10, 10)}}},
        "C": {0: {0: {300: (5, 11, 10)}}},
    }
    originals, hist_peaks = sf.find_original_alleles(tract_stats)
    assert originals[(0, 0)] == 5
    assert hist_peaks[(0, 0)] == 10


def test_find_original_alleles_multiple_alleles_and_tracts():
    tract_stats = {
        "A": {
            0: {0: {100: (5, 10, 20)}, 1: {100: (7, 15, 20)}},
            1: {0: {200: (3, 8, 25)}},
        },
        "B": {
            0: {0: {300: (5, 10, 15)}, 1: {300: (7, 15, 15)}},
        },
    }
    originals, _ = sf.find_original_alleles(tract_stats)
    assert originals[(0, 0)] == 5
    assert originals[(0, 1)] == 7
    assert originals[(1, 0)] == 3


# ---------------------------------------------------------------------------
# max_reads_fun
# ---------------------------------------------------------------------------

def test_max_reads_fun_picks_kd_with_highest_total_reads():
    tract_d = {
        100: (5, 10, 30),  # kd=5, 30 reads
        200: (7, 12, 100),  # kd=7, 100 reads
        300: (5, 10, 40),  # kd=5, 40 reads
    }
    # kd=5 has 30+40=70 total, kd=7 has 100. kd=7 wins.
    assert sf.max_reads_fun(tract_d) == 7


# ---------------------------------------------------------------------------
# make_determine_function
# ---------------------------------------------------------------------------

def test_make_determine_function_returns_original_when_available():
    original_lengths = {(0, 0): 5}
    fn = sf.make_determine_function(original_lengths)
    # Even though kd=7 has more reads, kd=5 matches original_lengths and should
    # be returned immediately.
    tract_d = {100: (7, 12, 100), 200: (5, 10, 10)}
    assert fn(tract_d, 0, 0) == 5


def test_make_determine_function_falls_back_to_max_reads_when_original_absent():
    original_lengths = {(0, 0): 99}  # not present in tract_d
    fn = sf.make_determine_function(original_lengths)
    tract_d = {100: (7, 12, 100), 200: (5, 10, 10)}
    # No tag tuple matches kd=99, so fall back to the most-reads kd (7).
    assert fn(tract_d, 0, 0) == 7


# ---------------------------------------------------------------------------
# find_most_likely_length
# ---------------------------------------------------------------------------

def test_find_most_likely_length_applies_determine_fn_to_every_tract():
    tract_stats = {
        "A": {
            0: {0: {100: (5, 10, 20)}, 1: {100: (7, 15, 20)}},
        },
    }
    original_lengths = {(0, 0): 5, (0, 1): 9}  # tract (0,1) -> not 7, so falls back
    fn = sf.make_determine_function(original_lengths)
    final = sf.find_most_likely_length(tract_stats, fn)
    assert final["A"][0][0] == 5
    assert final["A"][0][1] == 7  # fallback to kd with most reads


# ---------------------------------------------------------------------------
# get_fraction_less
# ---------------------------------------------------------------------------

def test_get_fraction_less_basic():
    # count_dict maps a value -> count. The function returns fraction of total
    # counts whose KEY is strictly greater than p.
    count_dict = {5: 10, 10: 20, 15: 30}
    # total = 60. keys > 5: {10:20, 15:30} = 50/60.
    assert sf.get_fraction_less(5, count_dict) == 50 / 60
    # keys > 10: {15:30} = 30/60 = 0.5.
    assert sf.get_fraction_less(10, count_dict) == 0.5
    # keys > 15: none = 0.
    assert sf.get_fraction_less(15, count_dict) == 0


# ---------------------------------------------------------------------------
# get_cells_with_match / get_cells_w_value
# ---------------------------------------------------------------------------

def _build_kd_df(rows):
    """Build a DataFrame with the MultiIndex test_peaks-style shape."""
    df = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "kd_allele"],
    )
    return df.set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])


def test_get_cells_with_match_returns_cells_with_consistent_kd():
    # Cell A: only kd=5 at (a=0,t=0) -> included
    # Cell B: kd=5 and kd=7 at (a=0,t=0) -> excluded (inconsistent)
    # Cell C: kd=5 at (a=0,t=0), kd=8 at (a=1,t=0) -> included
    rows = [
        ("A", "ins1", 0, 0, 100, 5),
        ("B", "ins1", 0, 0, 100, 5),
        ("B", "ins1", 0, 0, 200, 7),
        ("C", "ins1", 0, 0, 100, 5),
        ("C", "ins1", 1, 0, 200, 8),
    ]
    df = _build_kd_df(rows)
    matches = sf.get_cells_with_match(df, a_key=0, t_key=0, kd_allele=5)
    assert set(matches) == {"A", "C"}


def test_get_cells_w_value_returns_cells_with_non_null_kd():
    import numpy as np
    rows = [
        ("A", "ins1", 0, 0, 100, 5),
        ("B", "ins1", 0, 0, 100, np.nan),
        ("C", "ins1", 0, 0, 100, 7),
    ]
    df = _build_kd_df(rows)
    cells = sf.get_cells_w_value(df, a_key=0, t_key=0)
    assert set(cells) == {"A", "C"}


# ---------------------------------------------------------------------------
# get_mutated_cells_fun (second definition — overwrites the first)
# ---------------------------------------------------------------------------

def test_get_mutated_cells_fun_buckets_by_mutated_length():
    # zygote tract for (a=0, t=0) is 10.
    # Cell A: kd=12 (mutated to 12). Cell B: kd=10 (matches zygote). Cell C: kd=12.
    rows = [
        ("A", "ins1", 0, 0, 100, 12),
        ("A", "ins1", 0, 0, 200, 12),  # consistent kd within cell
        ("B", "ins1", 0, 0, 100, 10),  # matches zygote
        ("C", "ins1", 0, 0, 100, 12),
    ]
    df = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "kd_allele"],
    ).set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])

    zygote = pd.Series({(0, 0): 10})
    fn = sf.get_mutated_cells_fun(zygote)
    out = fn(df.reset_index())
    # Both A and C have a consistent kd of 12 that differs from zygote -> bucketed
    # under key (0, 0, 12).
    assert set(out[(0, 0, 12)]) == {"A", "C"}
    # B doesn't differ from zygote so it doesn't appear.
    assert "B" not in {c for cs in out.values() for c in cs}

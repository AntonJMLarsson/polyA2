"""Tests for build_insertion_summary."""
from collections import Counter

import numpy as np
import pandas as pd

import support_functions as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ins(chrom="chr1", start=100, end=200, strand="+", type_="KR"):
    return {"chrom": chrom, "start": start, "end": end, "strand": strand, "type": type_}


def _res_entry(allele_counts, cell_w_tag_site=250, cell_w_counts=300, allele_count_k=None):
    if allele_count_k is None:
        allele_count_k = {a: Counter({a: c}) for a, c in allele_counts.items()}
    return {
        "allele_counter": Counter(allele_counts),
        "tag_site_distribution_dict": {},
        "count_distribution_dict": {},
        "cell_w_counts": cell_w_counts,
        "cell_w_tag_site": cell_w_tag_site,
        "allele_count_n": {a: c for a, c in allele_counts.items()},
        "allele_count_k": allele_count_k,
    }


def _histogram_df(cells, ins_key="ins1"):
    """Minimal df with a MultiIndex containing a 'cell' level."""
    rows = [(c, ins_key, 0, 0, 100, 50) for c in cells]
    return pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "read_num"],
    ).set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_empty_insertion_information_returns_empty_df():
    df = sf.build_insertion_summary({}, {}, {}, {}, {}, {}, {}, {}, {})
    assert len(df) == 0


def test_minimal_row_when_not_in_any_dict():
    insertion_information = {"ins1": _ins()}
    df = sf.build_insertion_summary(
        insertion_information, {}, {}, {}, {}, {}, {}, {}, {}
    )
    assert df.loc["ins1", "chrom"] == "chr1"
    assert pd.isna(df.loc["ins1", "cell_w_tag_site"])
    assert bool(df.loc["ins1", "passed_cell_threshold"]) is False
    assert pd.isna(df.loc["ins1", "n_alleles_sequence"])
    assert pd.isna(df.loc["ins1", "allele_edit_distances"])


def test_res_dict_populates_cell_stats():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"A": 5, "B": 3}, cell_w_tag_site=150)}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, {}, {}, {}, {}, {}, {}, {}
    )
    assert df.loc["ins1", "cell_w_tag_site"] == 150
    assert df.loc["ins1", "n_allele_candidates"] == 2


def test_single_allele_no_biallelic_signal_marks_false():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"GTAC,ATCG": 50})}
    allele_calls = {"ins1": {"GTAC,ATCG": {"fraction_tags": 1.0}}}
    histogram = {"ins1": _histogram_df(["c1", "c2", "c3"])}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, histogram, {}, {}, {}, {}, {}
    )
    assert bool(df.loc["ins1", "passed_cell_threshold"]) is True
    assert df.loc["ins1", "n_alleles_sequence"] == 1
    assert df.loc["ins1", "allele_summaries"] == "GTAC,ATCG"
    # Single allele, no biallelic signal -> False (scanned but absent).
    assert bool(df.loc["ins1", "biallelic_signal_detected"]) is False
    assert df.loc["ins1", "n_cells_histogram"] == 3
    # Single allele -> no edit distances to compute.
    assert pd.isna(df.loc["ins1", "allele_edit_distances"])


def test_single_allele_with_biallelic_signal_marks_true():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AB,CD": 50})}
    allele_calls = {"ins1": {"AB,CD": {"fraction_tags": 1.0}}}
    histogram = {"ins1": _histogram_df(["c1"])}
    # Mimic the (boundaries_dict, _, _) shape returned by scan_for_biallelic_signal.
    boundaries_dict = {
        0: {"biallelic_peaks": np.array([10, 20]),
            "biallelic_boundaries": {10: 0, 20: 1},
            "min_peak": 10, "difference": 10},
    }
    peak_based = {"ins1": (boundaries_dict, None, None)}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, histogram, peak_based,
        {}, {}, {}, {},
    )
    assert bool(df.loc["ins1", "biallelic_signal_detected"]) is True
    assert "0:10,20" in df.loc["ins1", "biallelic_peaks"]
    assert df.loc["ins1", "biallelic_t_keys"] == "0"


def test_multi_allele_populates_pairwise_edit_distances():
    insertion_information = {"ins1": _ins()}
    # Two alleles with same tract count -> "seq=N" format.
    a1 = "GTAC,ATCG"
    a2 = "GTAC,ATCC"
    res_dict = {"ins1": _res_entry({a1: 50, a2: 40})}
    allele_calls = {
        "ins1": {
            a1: {"fraction_tags": 0.55, "conditional_probs": {a2: 0.01}},
            a2: {"fraction_tags": 0.45, "conditional_probs": {a1: 0.01}},
        }
    }
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, {}, {}, {}
    )
    assert df.loc["ins1", "n_alleles_sequence"] == 2
    # Sequence differs by one char ("G" -> "C") so seq edit dist = 1, same tracts.
    assert df.loc["ins1", "allele_edit_distances"] == "0-1:seq=1"


def test_multi_allele_structural_diff_includes_tracts_label():
    insertion_information = {"ins1": _ins()}
    a1 = "GTAC,ATCG"          # 1 tract
    a2 = "GTAC,XYZW,ATCG"     # 2 tracts (one extra in the middle)
    res_dict = {"ins1": _res_entry({a1: 50, a2: 40})}
    allele_calls = {
        "ins1": {
            a1: {"fraction_tags": 0.55, "conditional_probs": {a2: 0.01}},
            a2: {"fraction_tags": 0.45, "conditional_probs": {a1: 0.01}},
        }
    }
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, {}, {}, {}
    )
    val = df.loc["ins1", "allele_edit_distances"]
    assert "tracts=1" in val
    assert "seq=4" in val  # length of inserted XYZW


def test_three_alleles_produces_three_pairwise_distances():
    insertion_information = {"ins1": _ins()}
    a1, a2, a3 = "AA,BB", "AA,CC", "AA,DD"
    res_dict = {"ins1": _res_entry({a1: 50, a2: 40, a3: 30})}
    allele_calls = {
        "ins1": {
            a1: {"fraction_tags": 0.4, "conditional_probs": {a2: 0, a3: 0}},
            a2: {"fraction_tags": 0.35, "conditional_probs": {a1: 0, a3: 0}},
            a3: {"fraction_tags": 0.25, "conditional_probs": {a1: 0, a2: 0}},
        }
    }
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, {}, {}, {}
    )
    val = df.loc["ins1", "allele_edit_distances"]
    # Three pairwise comparisons, each separated by ';'.
    assert len(val.split(";")) == 3
    assert "0-1:" in val
    assert "0-2:" in val
    assert "1-2:" in val


def test_het_sites_and_snp_cpm_results():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AA,BB": 50})}
    allele_calls = {"ins1": {"AA,BB": {"fraction_tags": 1.0}}}
    het_sites = {"ins1": [[500, "A", "G", 0.5, 0.5], [600, "C", "T", 0.5, 0.5]]}
    # snp_cpm: variants_phased dict and unequal_len bool
    variants_phased = {500: {123: {"AA,BB": "A"}}, 600: {}}
    snp_cpm = {"ins1": (variants_phased, False, {})}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {},
        het_sites, snp_cpm, {}, {},
    )
    assert df.loc["ins1", "n_het_sites"] == 2
    assert df.loc["ins1", "snp_cpm_n_phased_variants"] == 1
    assert bool(df.loc["ins1", "snp_cpm_unequal_len"]) is False


def test_snp_cpm_early_return_handled():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AA,BB": 50})}
    allele_calls = {"ins1": {"AA,BB": {"fraction_tags": 1.0}}}
    # get_L1_flank_to_snp_cpm's early-return form: (None, None).
    snp_cpm = {"ins1": (None, None)}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, snp_cpm, {}, {}
    )
    assert df.loc["ins1", "snp_cpm_n_phased_variants"] == 0
    assert pd.isna(df.loc["ins1", "snp_cpm_unequal_len"])


def test_phase_results_pick_min_p_value():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AA,BB": 50, "AA,CC": 30})}
    allele_calls = {
        "ins1": {
            "AA,BB": {"fraction_tags": 0.6, "conditional_probs": {"AA,CC": 0}},
            "AA,CC": {"fraction_tags": 0.4, "conditional_probs": {"AA,BB": 0}},
        }
    }
    # Two het sites with different p-values.
    phase_allele = {
        "ins1": {
            500: ("counts", 12.5, 0.001, 1, "exp"),
            600: ("counts", 3.0, 0.08, 1, "exp"),
        }
    }
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, {},
        phase_allele, {},
    )
    assert df.loc["ins1", "phase_allele_snv_min_p"] == 0.001
    assert "500" in df.loc["ins1", "phase_allele_snv_sites"]
    assert "600" in df.loc["ins1", "phase_allele_snv_sites"]


def test_biallele_phase_results_recorded_separately():
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AA,BB": 50})}
    allele_calls = {"ins1": {"AA,BB": {"fraction_tags": 1.0}}}
    phase_biallele = {"ins1": {700: ("counts", 8.0, 0.005, 1, "exp")}}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, {}, {},
        phase_biallele,
    )
    assert df.loc["ins1", "phase_biallele_snv_min_p"] == 0.005
    assert "700" in df.loc["ins1", "phase_biallele_snv_sites"]


def test_below_threshold_has_none_for_biallelic_signal_field():
    # Not in allele_calls -> biallelic_signal_detected is None (not applicable),
    # not False.
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AA,BB": 5}, cell_w_tag_site=50)}
    df = sf.build_insertion_summary(
        insertion_information, res_dict, {}, {}, {}, {}, {}, {}, {}
    )
    assert df.loc["ins1", "biallelic_signal_detected"] is None or pd.isna(
        df.loc["ins1", "biallelic_signal_detected"]
    )


def test_multi_allele_biallelic_signal_field_is_none():
    # Multi-allele insertions skip biallelic scan -> field is None (not applicable).
    insertion_information = {"ins1": _ins()}
    res_dict = {"ins1": _res_entry({"AA,BB": 50, "AA,CC": 30})}
    allele_calls = {
        "ins1": {
            "AA,BB": {"fraction_tags": 0.6, "conditional_probs": {"AA,CC": 0}},
            "AA,CC": {"fraction_tags": 0.4, "conditional_probs": {"AA,BB": 0}},
        }
    }
    df = sf.build_insertion_summary(
        insertion_information, res_dict, allele_calls, {}, {}, {}, {}, {}, {}
    )
    val = df.loc["ins1", "biallelic_signal_detected"]
    assert val is None or pd.isna(val)

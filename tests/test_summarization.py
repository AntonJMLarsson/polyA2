"""Tests for sequence summarization and lightweight utility functions."""
from collections import Counter

import support_functions as sf


# ---------------------------------------------------------------------------
# summarize_sequence
# ---------------------------------------------------------------------------

def test_summarize_sequence_no_tract():
    # No A run reaches min_len, so we get a single trailing segment trimmed to cutoff.
    assert sf.summarize_sequence("GGGGCCCC", 5, 6, 0) == ["GGGGC"]


def test_summarize_sequence_single_tract_in_middle():
    # 6 As are enough to enter A_tract_state. The trailing non-A 'TCG'
    # immediately ends the tract because break_len=0.
    assert sf.summarize_sequence("GTACAAAAAATCG", 5, 6, 0) == ["GTAC", "TCG"]


def test_summarize_sequence_two_tracts():
    assert sf.summarize_sequence("GGAAAAAATTAAAAAACC", 5, 6, 0) == ["GG", "TT", "CC"]


def test_summarize_sequence_trailing_tract():
    # When the tract is at the end, no non-A flushes it. The trailing pre_string
    # is whatever was left before entering tract state (the trimmed flank).
    out = sf.summarize_sequence("GGGGAAAAAA", 5, 6, 0)
    assert out[0] == "GGGG"
    assert len(out) == 2


def test_summarize_sequence_below_min_len():
    # 5 As < min_len=6, so no tract is detected. The whole string is one flank,
    # truncated to cutoff.
    assert sf.summarize_sequence("CCCCAAAAATTTT", 5, 6, 0) == ["CCCCA"]


def test_summarize_sequence_cutoff_truncates_final_segment_only():
    # Intermediate segments aren't truncated; only the trailing flank.
    out = sf.summarize_sequence("XXXXXXXXXXAAAAAAYYYYYYYYYY", 4, 6, 0)
    assert out[0] == "XXXXXXXXXX"
    assert out[1] == "YYYY"


def test_summarize_sequence_break_len_bridges_interrupted_tract():
    # A single non-A in the middle of As gets bridged when break_len=1.
    # The whole stretch should look like one fused tract -> two flanks.
    out = sf.summarize_sequence("CCAAAAAATAAAAAAGG", 5, 6, 1)
    assert out[0] == "CC"
    # The trailing flank should be 'GG' (potentially with some pre-As stripped).
    assert out[-1].rstrip("A").endswith("GG") or out[-1] == "GG"


# ---------------------------------------------------------------------------
# summarize_sequence_and_count_A_tracts
# ---------------------------------------------------------------------------

def test_summarize_and_count_one_tract():
    parts, counts = sf.summarize_sequence_and_count_A_tracts(
        "GTACAAAAAATCG", 5, 6, 0
    )
    assert parts == ["GTAC", "TCG"]
    # Six As detected as a tract; trailing A_counter is 0 (no incomplete tract).
    assert counts == [6, 0]


def test_summarize_and_count_two_tracts():
    parts, counts = sf.summarize_sequence_and_count_A_tracts(
        "GGAAAAAATTAAAAAACC", 5, 6, 0
    )
    assert parts == ["GG", "TT", "CC"]
    assert counts == [6, 6, 0]


def test_summarize_and_count_no_tract():
    parts, counts = sf.summarize_sequence_and_count_A_tracts("CCGGCC", 5, 6, 0)
    assert parts == ["CCGGC"]
    assert counts == [0]


# ---------------------------------------------------------------------------
# count_A_tracts (legacy helper — only counts >6 consecutive As followed by non-A)
# ---------------------------------------------------------------------------

def test_count_A_tracts_basic():
    assert sf.count_A_tracts("GTACAAAAAAATCG") == [7]


def test_count_A_tracts_threshold_exclusive():
    # Exactly 6 As — function checks `> 6`, so 6 As is NOT counted.
    assert sf.count_A_tracts("GTACAAAAAATCG") == []


def test_count_A_tracts_trailing_tract_not_counted():
    # A trailing tract (no non-A terminator) is silently dropped — known quirk.
    assert sf.count_A_tracts("GTACAAAAAAA") == []


def test_count_A_tracts_multiple():
    assert sf.count_A_tracts("AAAAAAATAAAAAAAATG") == [7, 8]


# ---------------------------------------------------------------------------
# get_A_tract_length_for_allele
# ---------------------------------------------------------------------------

def test_get_A_tract_length_for_allele_aggregates_by_position():
    seqs = ["GTACAAAAAAATCGAAAAAAAATAG", "GTACAAAAAAATCGAAAAAAATAG"]
    identical_reads = {seqs[0]: 3, seqs[1]: 2}
    result = sf.get_A_tract_length_for_allele(seqs, identical_reads, n_A_tracts=2)
    # First tract: 7 As, both reads => 3+2 = 5 observations
    assert result[0][7] == 5
    # Second tract: read 0 has 8 As, read 1 has 7 As
    assert result[1][8] == 3
    assert result[1][7] == 2


# ---------------------------------------------------------------------------
# hamming_distance_unequal
# ---------------------------------------------------------------------------

def test_hamming_distance_unequal_basic():
    assert sf.hamming_distance_unequal("ACGT", "ACGT") == 0
    assert sf.hamming_distance_unequal("ACGT", "ACGA") == 1
    assert sf.hamming_distance_unequal("AAAA", "TTTT") == 4


def test_hamming_distance_unequal_uses_shorter_length():
    # Compares only over the overlap of the two strings.
    assert sf.hamming_distance_unequal("ACGT", "AC") == 0
    assert sf.hamming_distance_unequal("ACGTXY", "ACAT") == 1


# ---------------------------------------------------------------------------
# reconstruct_bottleneck_sequence
# ---------------------------------------------------------------------------

def test_reconstruct_bottleneck_sequence():
    # Two flanks separated by one polyA tract of length 6.
    summary = "GG,TT"
    flank = "TTCCCC"
    A_tract_lengths = {0: 6}
    out = sf.reconstruct_bottleneck_sequence(summary, flank, A_tract_lengths)
    assert out == "GG" + "A" * 6 + "TTCCCC"


# ---------------------------------------------------------------------------
# find_cells_with_sequence
# ---------------------------------------------------------------------------

def test_find_cells_with_sequence():
    read_dict = {
        "cellA": ["seq1", "seq2"],
        "cellB": ["seq3"],
        "cellC": ["seq1", "seq3"],
    }
    assert set(sf.find_cells_with_sequence(read_dict, "seq1")) == {"cellA", "cellC"}
    assert sf.find_cells_with_sequence(read_dict, "seq_missing") == []


# ---------------------------------------------------------------------------
# group_identical_reads and group_sequences
# ---------------------------------------------------------------------------

def test_group_identical_reads_counts_occurrences():
    read_dict = {
        "cell1": ["AA", "BB", "AA"],
        "cell2": ["AA", "CC"],
    }
    out = sf.group_identical_reads(read_dict)
    assert out == {"AA": 3, "BB": 1, "CC": 1}


def test_group_sequences_clusters_by_summary():
    seq_with_tract = "GTACAAAAAATCG"
    seq_diff_flank = "GTACAAAAAACCG"
    identical_reads = {seq_with_tract: 5, seq_diff_flank: 2}
    summaries, counts = sf.group_sequences(identical_reads, min_len=6, break_len=0)
    # Each distinct sequence is its own summary (differs in trailing flank).
    assert sum(counts.values()) == 7
    # The summary keys correspond to the comma-joined output of summarize_sequence.
    s1 = ",".join(sf.summarize_sequence(seq_with_tract, 5, 6, 0))
    s2 = ",".join(sf.summarize_sequence(seq_diff_flank, 5, 6, 0))
    assert counts[s1] == 5
    assert counts[s2] == 2
    assert seq_with_tract in summaries[s1]

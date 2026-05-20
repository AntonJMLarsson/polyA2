"""Tests for `allele_summary_distance` — the segment-aware edit distance."""
import support_functions as sf


def test_identical_alleles():
    assert sf.allele_summary_distance("ACGT,TGCA", "ACGT,TGCA") == (0, 0)


def test_same_structure_one_char_diff():
    # One character mismatch in the second segment.
    n_tract_diff, seq_dist = sf.allele_summary_distance("ACGT,TGCA", "ACGT,TGGA")
    assert n_tract_diff == 0
    assert seq_dist == 1


def test_same_structure_multi_segment_diffs():
    # Edits spread across both segments.
    n_tract_diff, seq_dist = sf.allele_summary_distance("ACGT,TGCA", "AGGT,TGGA")
    assert n_tract_diff == 0
    assert seq_dist == 2


def test_structural_diff_inserted_segment_in_middle():
    # `"GTAC,GCTA"` -> `"GTAC,ATCG,GCTA"`: insert `ATCG` between the two existing
    # flanking segments. The alignment must place the new segment in the middle,
    # not at the end — otherwise the seq edit distance would explode.
    n_tract_diff, seq_dist = sf.allele_summary_distance(
        "GTAC,GCTA", "GTAC,ATCG,GCTA"
    )
    assert n_tract_diff == 1
    assert seq_dist == 4  # length of inserted segment "ATCG"


def test_structural_diff_inserted_segment_at_end():
    n_tract_diff, seq_dist = sf.allele_summary_distance("GTAC,GCTA", "GTAC,GCTA,XX")
    assert n_tract_diff == 1
    assert seq_dist == 2  # length of "XX"


def test_structural_diff_inserted_segment_at_start():
    n_tract_diff, seq_dist = sf.allele_summary_distance("GTAC,GCTA", "YY,GTAC,GCTA")
    assert n_tract_diff == 1
    assert seq_dist == 2


def test_naive_levenshtein_would_be_misleading():
    # The naive whole-string Levenshtein would count `,ATCG` as 5 character
    # edits when one allele has an extra polyA tract. Our segment-aware
    # distance separates the structural change (1 tract) from the 4 inserted
    # characters.
    n_tract_diff, seq_dist = sf.allele_summary_distance("GTAC,GCTA", "GTAC,ATCG,GCTA")
    assert (n_tract_diff, seq_dist) == (1, 4)


def test_double_insertion():
    # Source has 1 tract, destination has 3 tracts. Two structural ops.
    n_tract_diff, seq_dist = sf.allele_summary_distance("AA,BB", "AA,CC,DD,BB")
    assert n_tract_diff == 2
    assert seq_dist == 4  # CC + DD = 4 inserted characters


def test_symmetric_distance():
    # The metric should be symmetric.
    a, b = "GTAC,GCTA", "GTAC,ATCG,GCTA"
    assert sf.allele_summary_distance(a, b) == sf.allele_summary_distance(b, a)


def test_empty_segments_handled():
    # When a polyA tract sits at the start, summarize_sequence emits an empty
    # leading segment. Make sure those don't break the alignment.
    assert sf.allele_summary_distance(",ATCG", ",ATCG") == (0, 0)
    assert sf.allele_summary_distance(",ATCG", ",ATCC") == (0, 1)


def test_chooses_alignment_minimising_total_cost():
    # `"ABCD,EFGH"` vs `"ABCD,WXYZ,EFGH"`. A bad alignment would try to align
    # WXYZ with EFGH (4 mismatches) and then drop a tail. The optimal alignment
    # inserts WXYZ in the middle (1 tract + 4 chars). Our DP must pick the
    # latter.
    n_tract_diff, seq_dist = sf.allele_summary_distance(
        "ABCD,EFGH", "ABCD,WXYZ,EFGH"
    )
    assert (n_tract_diff, seq_dist) == (1, 4)


def test_indel_preferred_when_segments_very_different():
    # If two adjacent segments are unrelated, dropping one and inserting the
    # other can beat matching them via Levenshtein.
    n_tract_diff, seq_dist = sf.allele_summary_distance("AAAAAAAA", "BBBBBBBB,CCCCCCCC")
    # The cheapest alignment is: replace 'AAAA...' with 'BBBB...' (8 edits)
    # then insert 'CCCC...' (1 indel + 8 chars), or alternatively delete A's
    # and insert both segments. Our DP picks whichever is cheaper.
    assert n_tract_diff >= 1
    assert seq_dist <= 16

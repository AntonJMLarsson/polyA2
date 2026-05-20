"""Tests for the dict-based allele filtering and coupling helpers."""
from collections import Counter

import support_functions as sf


# ---------------------------------------------------------------------------
# get_correct_alleles
# ---------------------------------------------------------------------------

def test_get_correct_alleles_single_dominant():
    data = {"cell1": {100: Counter({"A": 15, "noise": 2})}}
    allele_counter, tag_dist, count_dist, cell_w_counts, cell_w_tag = (
        sf.get_correct_alleles(data)
    )
    assert allele_counter == Counter({"A": 1})
    assert tag_dist == {"A": Counter({100: 1})}
    assert count_dist == {"A": Counter({15: 1})}
    assert cell_w_counts == 1
    assert cell_w_tag == 1


def test_get_correct_alleles_below_count_threshold():
    # Sum <= 10 -> doesn't even count as "has_counts".
    data = {"cell1": {100: Counter({"A": 5, "B": 2})}}
    counter, _, _, cwc, cwt = sf.get_correct_alleles(data)
    assert counter == Counter()
    assert cwc == 0
    assert cwt == 0


def test_get_correct_alleles_has_counts_but_no_tag_site():
    # Sum > 10 but no single allele has > 10 reads.
    data = {"cell1": {100: Counter({"A": 6, "B": 6})}}
    counter, _, _, cwc, cwt = sf.get_correct_alleles(data)
    assert counter == Counter()
    assert cwc == 1
    assert cwt == 0


def test_get_correct_alleles_aggregates_multiple_cells():
    data = {
        "cell1": {100: Counter({"A": 15}), 200: Counter({"A": 12, "B": 2})},
        "cell2": {300: Counter({"B": 20})},
        "cell3": {400: Counter({"A": 11})},
    }
    counter, tag_dist, _, cwc, cwt = sf.get_correct_alleles(data)
    assert counter == Counter({"A": 3, "B": 1})
    assert tag_dist["A"] == Counter({100: 1, 200: 1, 400: 1})
    assert tag_dist["B"] == Counter({300: 1})
    assert cwc == 3
    assert cwt == 3


def test_get_correct_alleles_empty():
    counter, tag_dist, _, cwc, cwt = sf.get_correct_alleles({})
    assert counter == Counter()
    assert tag_dist == {}
    assert (cwc, cwt) == (0, 0)


# ---------------------------------------------------------------------------
# get_allele_coupling
# ---------------------------------------------------------------------------

def test_get_allele_coupling_tracks_co_observation():
    data = {
        "cell1": {
            100: Counter({"A": 15, "B": 3}),
            200: Counter({"A": 20, "B": 5}),
            300: Counter({"B": 12, "A": 1}),
        }
    }
    allele_counter = Counter({"A": 2, "B": 1})
    count_n, count_k = sf.get_allele_coupling(data, allele_counter)
    # Site 100,200: A wins -> n[A] gets 15+20=35. k[A][A] is also 35. k[A][B] = 3+5 = 8.
    # Site 300: B wins -> n[B] gets 12. k[B][B] = 12. k[B][A] = 1.
    assert count_n == {"A": 35, "B": 12}
    assert count_k["A"]["A"] == 35
    assert count_k["A"]["B"] == 8
    assert count_k["B"]["A"] == 1
    assert count_k["B"]["B"] == 12


def test_get_allele_coupling_ignores_sites_below_threshold():
    # Most-common count must exceed 10 to be tallied.
    data = {"cell1": {100: Counter({"A": 8, "B": 2})}}
    count_n, count_k = sf.get_allele_coupling(data, Counter({"A": 1, "B": 1}))
    assert count_n == {"A": 0, "B": 0}
    assert count_k["A"]["A"] == 0
    assert count_k["A"]["B"] == 0


def test_get_allele_coupling_ignores_unknown_alleles():
    # `noise` is not in allele_counter so it should not appear in count_k.
    data = {"cell1": {100: Counter({"A": 15, "noise": 4})}}
    count_n, count_k = sf.get_allele_coupling(data, Counter({"A": 1}))
    assert count_n == {"A": 15}
    assert count_k["A"] == Counter({"A": 15})


# ---------------------------------------------------------------------------
# compare_allele_stats
# ---------------------------------------------------------------------------

def test_compare_allele_stats_conditional_probs():
    info_dict = {
        "allele_count_k": {
            "A": Counter({"A": 100, "B": 5}),
            "B": Counter({"A": 2, "B": 80}),
        }
    }
    allele_key_dict = {"A": {}, "B": {}}
    out = sf.compare_allele_stats(info_dict, allele_key_dict)
    assert out["A"]["B"] == 0.05
    assert out["B"]["A"] == 0.025
    # The function only fills off-diagonal entries (permutations).
    assert "A" not in out["A"]


def test_compare_allele_stats_three_alleles():
    info_dict = {
        "allele_count_k": {
            "A": Counter({"A": 100, "B": 10, "C": 5}),
            "B": Counter({"A": 4, "B": 80, "C": 2}),
            "C": Counter({"A": 1, "B": 3, "C": 50}),
        }
    }
    allele_key_dict = {"A": {}, "B": {}, "C": {}}
    out = sf.compare_allele_stats(info_dict, allele_key_dict)
    assert out["A"]["B"] == 10 / 100
    assert out["A"]["C"] == 5 / 100
    assert out["B"]["A"] == 4 / 80
    assert out["B"]["C"] == 2 / 80
    assert out["C"]["A"] == 1 / 50
    assert out["C"]["B"] == 3 / 50

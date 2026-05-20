"""Tests for `correction_to_apply` — how an action list maps to per-tract
length adjustments on the destination allele."""
import support_functions as sf


def test_empty_action_list_returns_zeros():
    # One comma in destination -> one tract -> one key.
    assert sf.correction_to_apply([], "AB,CD") == {0: 0}


def test_two_tracts_initial_zeros():
    assert sf.correction_to_apply([], "AB,CD,EF") == {0: 0, 1: 0}


def test_strip_right_adds_to_tract_at_action_index():
    # ('strip', i, len, 'r'): characters stripped from the right of segment i
    # add to the tract that follows segment i (index i).
    assert sf.correction_to_apply(
        [("strip", 0, 2, "r")], "AB,CD"
    ) == {0: 2}


def test_strip_right_ignored_when_index_equals_n_tracts():
    # `action[1] == len(correction_dict)` -> strip is on the trailing flank
    # and there's no tract to attribute it to. The function skips it.
    assert sf.correction_to_apply(
        [("strip", 1, 3, "r")], "AB,CD"
    ) == {0: 0}


def test_strip_left_adds_to_previous_tract():
    # ('strip', i, len, 'l'): characters stripped from the left of segment i
    # belong to the tract preceding it (index i-1).
    assert sf.correction_to_apply(
        [("strip", 1, 2, "l")], "AB,CD,EF"
    ) == {0: 2, 1: 0}


def test_prepend_decreases_previous_tract():
    # ('prepend', i, 0, seq): seq characters added in front of segment i
    # come out of the preceding tract (index i-1).
    assert sf.correction_to_apply(
        [("prepend", 1, 0, "XYZ")], "AB,CD,EF"
    ) == {0: -3, 1: 0}


def test_append_decreases_following_tract():
    # ('append', i, 0, seq): seq added after segment i comes out of the
    # following tract (index i).
    assert sf.correction_to_apply(
        [("append", 0, 0, "XY")], "AB,CD"
    ) == {0: -2}


def test_append_ignored_when_index_equals_n_tracts():
    # The trailing flank has no following tract — same skip as strip 'r'.
    assert sf.correction_to_apply(
        [("append", 1, 0, "Z")], "AB,CD"
    ) == {0: 0}


def test_combined_actions_sum():
    # Two tracts (indices 0, 1) for destination "AB,CD,EF".
    # strip 'r' at segment 0 -> +1 on tract 0
    # prepend at segment 1   -> -1 on tract 0
    # append at segment 1    -> -2 on tract 1 (segment 1 is the middle, not trailing)
    actions = [
        ("strip", 0, 1, "r"),
        ("prepend", 1, 0, "X"),
        ("append", 1, 0, "YZ"),
    ]
    assert sf.correction_to_apply(actions, "AB,CD,EF") == {0: 0, 1: -2}


def test_insert_action_returns_none():
    assert sf.correction_to_apply(
        [("insert", 0, 0, "X")], "AB,CD"
    ) is None


def test_remove_action_returns_none():
    assert sf.correction_to_apply(
        [("remove", 0, 0, "")], "AB,CD"
    ) is None


def test_abort_action_returns_none():
    assert sf.correction_to_apply(
        [("abort", 0, 0, 0)], "AB,CD"
    ) is None


def test_infinite_action_returns_none():
    assert sf.correction_to_apply(
        [("infinite", 0, 0, 0)], "AB,CD"
    ) is None


def test_any_action_field_matching_abort_returns_none():
    # The abort check matches *any* tuple element against {insert, abort,
    # infinite, remove}, so an inner 'abort' marker in a non-abort action
    # still triggers the bail-out.
    assert sf.correction_to_apply(
        [("strip", 0, 1, "abort")], "AB,CD"
    ) is None

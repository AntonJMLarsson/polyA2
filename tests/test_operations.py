"""Tests for the operation list / action machinery used by
`calculate_minimum_operations` to transform one allele summary into another.
"""
import support_functions as sf


# ---------------------------------------------------------------------------
# get_op_seq
# ---------------------------------------------------------------------------

def test_get_op_seq_keeps_leading_char_then_only_As():
    # Always keeps index 0; then keeps consecutive As until a non-A is hit.
    assert sf.get_op_seq("CAAAT") == "CAAA"
    assert sf.get_op_seq("CTAGC") == "C"
    assert sf.get_op_seq("AAAA") == "AAAA"
    assert sf.get_op_seq("A") == "A"
    assert sf.get_op_seq("") == ""


# ---------------------------------------------------------------------------
# get_operation_list
# ---------------------------------------------------------------------------

def test_get_operation_list_two_segments():
    out = sf.get_operation_list("AB,CD")
    # First segment: reverse "AB" -> "BA"; op_seq("BA")="BA" (B then A); reversed back -> "AB".
    # Last segment: op_seq("CD") = "C".
    assert out == [("AB", "AB"), ("CD", "C")]


def test_get_operation_list_three_segments():
    out = sf.get_operation_list("AB,CD,EF")
    # Middle segment gets a (begin, end) tuple of leading-and-trailing op_seqs.
    assert out[0] == ("AB", "AB")
    assert out[1] == ("CD", ("C", "D"))
    assert out[2] == ("EF", "E")


def test_get_operation_list_single_segment():
    # Single segment: i == n-1 path -> ('seq', get_op_seq(seq))
    # But for n==1, i==0 is also the only iteration; the function only hits the
    # i==0 branch which is the "first segment" path (reverse and back).
    out = sf.get_operation_list("CTAGC")
    assert out == [("CTAGC", "C")]


# ---------------------------------------------------------------------------
# Individual action functions
# ---------------------------------------------------------------------------

def test_edit_action_replaces_char_at_index():
    assert sf.edit_action(["AB", "CD"], ("edit", 0, 1, "Z")) == ["AZ", "CD"]


def test_prepend_action_j_zero_prepends_full_seq():
    assert sf.prepend_action(["AB", "CD"], ("prepend", 1, 0, "XY")) == ["AB", "XYCD"]


def test_prepend_action_j_nonzero_truncates_after_prepend():
    # j != 0: source becomes (seq + source)[:j]
    out = sf.prepend_action(["AB", "CDE"], ("prepend", 1, 2, "XY"))
    assert out == ["AB", "XY"]


def test_append_action_concatenates():
    assert sf.append_action(["AB", "CD"], ("append", 1, 0, "X")) == ["AB", "CDX"]


def test_strip_action_right_keeps_first_j_chars():
    # NB: strip_action with seq='r' uses j as a slice index, so source[:j]
    # keeps the first j characters and drops the rest.
    assert sf.strip_action(["ABCDE"], ("strip", 0, 2, "r")) == ["AB"]


def test_strip_action_left_drops_first_j_chars():
    assert sf.strip_action(["ABCDE"], ("strip", 0, 2, "l")) == ["CDE"]


def test_insert_action_inserts_at_index():
    assert sf.insert_action(["A", "B"], ("insert", 1, 0, "X")) == ["A", "X", "B"]
    assert sf.insert_action(["A", "B"], ("insert", 0, 0, "X")) == ["X", "A", "B"]


def test_remove_action_drops_element_at_index():
    assert sf.remove_action(["A", "B", "C"], ("remove", 1, 0, "")) == ["A", "C"]


def test_truncate_action_keeps_first_i_elements():
    assert sf.truncate_action(["A", "B", "C", "D"], ("truncate", 2, 0, "")) == ["A", "B"]
    assert sf.truncate_action(["A", "B", "C"], ("truncate", 0, 0, "")) == []


# ---------------------------------------------------------------------------
# apply_action_list dispatches through the `allowed_actions` registry.
# ---------------------------------------------------------------------------

def test_apply_action_list_chains_actions():
    # Start with ["AB", "CD"], edit the first segment, then append to the last.
    out = sf.apply_action_list(
        ["AB", "CD"], [("edit", 0, 0, "Z"), ("append", 1, 0, "X")]
    )
    assert out == ["ZB", "CDX"]


def test_apply_action_list_insert_then_remove():
    out = sf.apply_action_list(
        ["A", "B"], [("insert", 1, 0, "X"), ("remove", 0, 0, "")]
    )
    assert out == ["X", "B"]


# ---------------------------------------------------------------------------
# calculate_minimum_operations
# ---------------------------------------------------------------------------

def test_calculate_minimum_operations_identical_is_empty():
    # No edits needed: the action list comes back empty.
    assert sf.calculate_minimum_operations("AB,CD", "AB", "AB,CD", "AB") == []


def test_calculate_minimum_operations_insert_when_source_shorter():
    # Source has 1 segment, destination has 2 — must insert a segment.
    actions = sf.calculate_minimum_operations("AB", "AB", "AB,CD", "CD")
    # First action should be an insert.
    assert actions[0][0] == "insert"


def test_calculate_minimum_operations_remove_when_source_longer():
    # Source has 2 segments, destination has 1 — must remove a segment.
    actions = sf.calculate_minimum_operations("AB,CD", "CD", "AB", "AB")
    assert actions[0][0] == "remove"

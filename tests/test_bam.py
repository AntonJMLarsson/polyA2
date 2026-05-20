"""Integration tests that exercise the BAM-fetching pipeline against a
synthetic BAM file generated inside a pytest fixture.

Notes on the test design:
- The insertion is on the '+' strand. Only reverse-strand reads pass
  `correct_orientation` for '+', so all synthetic reads are flagged is_reverse.
- Tagmentation site for reverse reads = reference_end, so ref_start + 50.
- LS tag is what summarize_sequence reads — independent of the aligned bases
  (query_sequence). We can control SNPs (via query_sequence) and allele identity
  (via LS) independently.
"""
import os
from collections import Counter

import pysam
import pytest

import support_functions as sf


# Two distinct sequences that produce different polyA-tract summaries.
# LS_A = "GTACAAAAAAATCG"  ->  summary "GTAC,TCG"
# LS_B = "ACGTAAAAAAATAG"  ->  summary "ACGT,TAG"
LS_A = "GTACAAAAAAATCG"
LS_B = "ACGTAAAAAAATAG"
SUMMARY_A = "GTAC,TCG"
SUMMARY_B = "ACGT,TAG"
INSERTION = {"chrom": "chr1", "start": 400, "end": 700, "strand": "+", "type": "KR"}


def _write_read(bam, name, cell, ls, query_seq, ref_start, length=50, qual=40):
    """Write one reverse-strand read with BC and LS tags."""
    a = pysam.AlignedSegment()
    a.query_name = name
    a.query_sequence = query_seq
    a.flag = 16  # reverse strand
    a.reference_id = 0
    a.reference_start = ref_start
    a.cigar = [(0, length)]  # M{length}
    a.mapping_quality = 60
    a.query_qualities = pysam.qualitystring_to_array(chr(qual + 33) * length)
    a.set_tags([("BC", cell, "Z"), ("LS", ls, "Z")])
    bam.write(a)


def _make_bam(path, reads):
    """reads = list of (name, cell, ls, query_seq, ref_start)."""
    header = {"HD": {"VN": "1.0", "SO": "coordinate"},
              "SQ": [{"LN": 10_000, "SN": "chr1"}]}
    # Sort by ref_start so the BAM is coordinate-sorted (required for indexing).
    reads = sorted(reads, key=lambda r: r[4])
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for r in reads:
            _write_read(bam, *r)
    pysam.index(str(path))


# ---------------------------------------------------------------------------
# Fixture: small BAM exercising allele detection + per-cell tag-site grouping.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_bam(tmp_path_factory):
    """Two cells, two alleles, multiple tagmentation sites.

    CELL_A: 15 reads of allele A at tag_site=500, plus 3 noise reads of allele B
    CELL_B: 12 reads of allele B at tag_site=520
    CELL_C: 14 reads of allele A at tag_site=540
    UNKNOWN: 5 reads — outside the cell set, must be ignored
    """
    path = tmp_path_factory.mktemp("bam") / "small.bam"
    qseq = "N" * 50
    reads = []
    for i in range(15):
        reads.append((f"a{i}", "CELL_A", LS_A, qseq, 500 - 50))  # ref_end=500
    for i in range(3):
        reads.append((f"an{i}", "CELL_A", LS_B, qseq, 500 - 50))
    for i in range(12):
        reads.append((f"b{i}", "CELL_B", LS_B, qseq, 520 - 50))
    for i in range(14):
        reads.append((f"c{i}", "CELL_C", LS_A, qseq, 540 - 50))
    for i in range(5):
        reads.append((f"u{i}", "UNKNOWN", LS_A, qseq, 500 - 50))
    _make_bam(path, reads)
    return str(path)


def test_collect_tag_site_info_groups_per_cell_and_tag_site(small_bam):
    cell_set = {"CELL_A", "CELL_B", "CELL_C"}
    out = sf.collect_tag_site_info(small_bam, cell_set, INSERTION, min_len=6, break_len=0)

    # The unknown cell should not appear among populated entries.
    populated = {c: d for c, d in out.items() if d}
    assert set(populated.keys()) == {"CELL_A", "CELL_B", "CELL_C"}

    # CELL_A has tag_site=500 with both allele summaries.
    a_dict = out["CELL_A"][500]
    assert a_dict[SUMMARY_A] == 15
    assert a_dict[SUMMARY_B] == 3

    # CELL_B has tag_site=520 with allele B dominant.
    assert out["CELL_B"][520][SUMMARY_B] == 12


def test_collect_tag_site_info_ignores_wrong_orientation(tmp_path):
    # Build a BAM with forward-strand reads; '+' insertion should reject them.
    path = tmp_path / "fwd.bam"
    qseq = "N" * 50
    header = {"HD": {"VN": "1.0", "SO": "coordinate"},
              "SQ": [{"LN": 10_000, "SN": "chr1"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for i in range(10):
            a = pysam.AlignedSegment()
            a.query_name = f"f{i}"
            a.query_sequence = qseq
            a.flag = 0  # forward strand
            a.reference_id = 0
            a.reference_start = 450
            a.cigar = [(0, 50)]
            a.mapping_quality = 60
            a.query_qualities = pysam.qualitystring_to_array("I" * 50)
            a.set_tags([("BC", "CELL_A", "Z"), ("LS", LS_A, "Z")])
            bam.write(a)
    pysam.index(str(path))

    out = sf.collect_tag_site_info(str(path), {"CELL_A"}, INSERTION, min_len=6, break_len=0)
    # No reads should have been collected because none are reverse-strand.
    assert out["CELL_A"] == {}


def test_get_alleles_by_tagmentation_site_end_to_end(small_bam):
    cell_set = {"CELL_A", "CELL_B", "CELL_C"}
    key, result = sf.get_alleles_by_tagmentation_site(
        small_bam, cell_set, INSERTION, "ins1", min_len=6, break_len=0
    )
    assert key == "ins1"
    # Two alleles should be detected (allele A wins in CELL_A and CELL_C; allele B in CELL_B).
    assert result["allele_counter"][SUMMARY_A] == 2  # 2 cells voted for A
    assert result["allele_counter"][SUMMARY_B] == 1  # 1 cell voted for B
    # cell_w_tag_site counts cells with at least one >10-read site.
    assert result["cell_w_tag_site"] == 3
    # allele_count_k tracks co-observed counts at sites where the column allele wins.
    # CELL_A site 500: A=15 (winner), B=3 noise -> k[A][A] += 15, k[A][B] += 3
    # CELL_C site 540: A=14 (winner) -> k[A][A] += 14
    assert result["allele_count_k"][SUMMARY_A][SUMMARY_A] == 29
    assert result["allele_count_k"][SUMMARY_A][SUMMARY_B] == 3


# ---------------------------------------------------------------------------
# Fixture: BAM for SNP-level tests. All reads at the same tag site with the
# same LS (one allele) but heterozygous bases at position 480.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def snp_bam(tmp_path_factory):
    """50 reads with 'A' at pos 480, 50 with 'G'. All ref_start=450, length=50.
    Position 480 corresponds to query_alignment_sequence index 30."""
    path = tmp_path_factory.mktemp("bam") / "snp.bam"
    reads = []
    base_seq = list("N" * 50)
    for i in range(50):
        qs = base_seq.copy()
        qs[30] = "A"
        reads.append((f"sa{i}", "CELL_A", LS_A, "".join(qs), 450))
    for i in range(50):
        qs = base_seq.copy()
        qs[30] = "G"
        reads.append((f"sg{i}", "CELL_B", LS_A, "".join(qs), 450))
    _make_bam(path, reads)
    return str(path)


def test_get_allele_to_snp_counts(snp_bam):
    cell_set = {"CELL_A", "CELL_B"}
    het_site = [480, "A", "G", 0.5, 0.5]
    allele_call = {SUMMARY_A: {"fraction_tags": 1.0}}
    out = sf.get_allele_to_snp_counts(
        snp_bam, cell_set, INSERTION, allele_call, het_site, "ins1",
        min_len=6, break_len=0,
    )
    # All 100 reads share the same allele summary; SNPs split 50/50.
    assert out[SUMMARY_A]["A"] == 50
    assert out[SUMMARY_A]["G"] == 50


@pytest.fixture(scope="module")
def phase_bam(tmp_path_factory):
    """Two distinct allele summaries, each carrying its own SNV variant at
    position 480: allele A reads are 'A', allele B reads are 'G'."""
    path = tmp_path_factory.mktemp("bam") / "phase.bam"
    reads = []
    base = list("N" * 50)
    for i in range(40):
        qs = base.copy()
        qs[30] = "A"
        reads.append((f"a{i}", "CELL_A", LS_A, "".join(qs), 450))
    for i in range(40):
        qs = base.copy()
        qs[30] = "G"
        reads.append((f"b{i}", "CELL_B", LS_B, "".join(qs), 450))
    _make_bam(path, reads)
    return str(path)


def test_phase_L1_and_flank_SNV_runs_chi2(phase_bam):
    cell_set = {"CELL_A", "CELL_B"}
    allele_call = {
        SUMMARY_A: {"fraction_tags": 0.5},
        SUMMARY_B: {"fraction_tags": 0.5},
    }
    het_sites = [[480, "A", "G", 0.5, 0.5]]
    key, out = sf.phase_L1_and_flank_SNV(
        phase_bam, cell_set, INSERTION, allele_call, het_sites, "ins1",
        min_len=6, break_len=0,
    )
    assert key == "ins1"
    allele_to_snp, chi2, p, dof, expected = out[480]
    # Perfect linkage: allele A reads are 'A', allele B reads are 'G'.
    assert allele_to_snp[SUMMARY_A]["A"] == 40
    assert allele_to_snp[SUMMARY_A]["G"] == 0
    assert allele_to_snp[SUMMARY_B]["A"] == 0
    assert allele_to_snp[SUMMARY_B]["G"] == 40
    assert p < 0.001


# ---------------------------------------------------------------------------
# Biallelic-signal phasing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def biallele_bam(tmp_path_factory):
    """One allele summary (SUMMARY_A), two cells each with their own tagmentation
    site. Cell A reads carry 'A' at the het site; Cell B reads carry 'G'."""
    path = tmp_path_factory.mktemp("bam") / "biallele.bam"
    reads = []
    base = list("N" * 50)
    # Cell A: 30 reads at tag_site=500 (ref_end=500), position 480 has 'A'.
    for i in range(30):
        qs = base.copy()
        qs[30] = "A"  # pos 480 = ref_start + 30
        reads.append((f"a{i}", "CELL_A", LS_A, "".join(qs), 450))
    # Cell B: 30 reads at tag_site=600 (ref_end=600), position 580 has 'G'.
    for i in range(30):
        qs = base.copy()
        qs[30] = "G"
        reads.append((f"b{i}", "CELL_B", LS_A, "".join(qs), 550))
    _make_bam(path, reads)
    return str(path)


def test_get_biallele_to_snp_counts(biallele_bam):
    """Build a synthetic boundaries_dict + test_peaks that map CELL_A's tag site
    to biallele 0 and CELL_B's to biallele 1, then verify SNP counts respect the
    mapping."""
    import pandas as pd

    allele_call = {SUMMARY_A: {"fraction_tags": 1.0}}
    # Place CELL_A reads at hist_peak=10 (biallele 0), CELL_B at hist_peak=20 (biallele 1).
    rows = [
        ("CELL_A", "ins1", 0, 0, 500, 10, 50),
        ("CELL_B", "ins1", 0, 0, 600, 20, 50),
    ]
    test_peaks = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "hist_peak", "read_num"],
    ).set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])
    boundaries_dict = {
        0: {"biallelic_peaks": [10, 20],
            "biallelic_boundaries": {10: 0, 20: 1},
            "min_peak": 10, "difference": 10}
    }

    # Het site at position 480 (covered by CELL_A reads, ref_start=450) AND
    # at position 580 (covered by CELL_B reads, ref_start=550).
    # The function fetches around het_site[0]±1, so test each one separately.
    out_a = sf.get_biallele_to_snp_counts(
        biallele_bam, {"CELL_A", "CELL_B"}, INSERTION, allele_call,
        [480, "A", "G", 0.5, 0.5], test_peaks, boundaries_dict,
        min_len=6, break_len=0,
    )
    # Only CELL_A reads cover position 480; they're all biallele 0 with 'A'.
    assert out_a[0]["A"] == 30
    assert out_a[0]["G"] == 0
    assert out_a[1]["A"] == 0  # no CELL_B reads cover 480

    out_b = sf.get_biallele_to_snp_counts(
        biallele_bam, {"CELL_A", "CELL_B"}, INSERTION, allele_call,
        [580, "A", "G", 0.5, 0.5], test_peaks, boundaries_dict,
        min_len=6, break_len=0,
    )
    # Only CELL_B reads cover position 580; they're all biallele 1 with 'G'.
    assert out_b[1]["G"] == 30
    assert out_b[1]["A"] == 0
    assert out_b[0]["G"] == 0


@pytest.fixture(scope="module")
def shared_position_bam(tmp_path_factory):
    """Both cells' reads cover the SAME genomic position so we can test
    chi-square on a real het site between bialleles."""
    path = tmp_path_factory.mktemp("bam") / "shared.bam"
    reads = []
    base = list("N" * 50)
    # CELL_A: 30 reads at tag_site=500 (ref_start=450). Pos 480 has 'A'.
    for i in range(30):
        qs = base.copy()
        qs[30] = "A"
        reads.append((f"a{i}", "CELL_A", LS_A, "".join(qs), 450))
    # CELL_B: 30 reads at tag_site=500 too (different cell, same tag_site). Pos 480 has 'G'.
    for i in range(30):
        qs = base.copy()
        qs[30] = "G"
        reads.append((f"b{i}", "CELL_B", LS_A, "".join(qs), 450))
    _make_bam(path, reads)
    return str(path)


def test_phase_biallelic_signal_at_shared_position(shared_position_bam):
    import pandas as pd

    allele_call = {SUMMARY_A: {"fraction_tags": 1.0}}
    # Place CELL_A and CELL_B at the SAME tag_site (500) but with different
    # hist_peaks so they fall into different bialleles.
    rows = [
        ("CELL_A", "ins1", 0, 0, 500, 10, 50),
        ("CELL_B", "ins1", 0, 0, 500, 20, 50),
    ]
    test_peaks = pd.DataFrame(
        rows,
        columns=["cell", "ins_key", "a_key", "t_key", "tag_site", "hist_peak", "read_num"],
    ).set_index(["cell", "ins_key", "a_key", "t_key", "tag_site"])
    # Wait — both cells share tag_site=500, so the (cell, tag_site) key is unique
    # per cell. The boundaries_dict maps hist_peak -> biallele.
    boundaries_dict = {
        0: {"biallelic_peaks": [10, 20],
            "biallelic_boundaries": {10: 0, 20: 1},
            "min_peak": 10, "difference": 10}
    }
    biallele_result = (boundaries_dict, None, None)

    key, out = sf.phase_biallelic_signal_and_SNV(
        shared_position_bam, {"CELL_A", "CELL_B"}, INSERTION, allele_call,
        [[480, "A", "G", 0.5, 0.5]], "ins1", test_peaks, biallele_result,
        min_len=6, break_len=0,
    )
    assert key == "ins1"
    counts, chi2, p, dof, expected = out[480]
    # Perfect concordance: biallele 0 = all A, biallele 1 = all G.
    assert counts[0]["A"] == 30
    assert counts[0]["G"] == 0
    assert counts[1]["A"] == 0
    assert counts[1]["G"] == 30
    # chi2 should be large and p very small for perfect concordance.
    assert chi2 is not None
    assert p < 0.001


# ---------------------------------------------------------------------------
# Heterozygous position detection — needs >7000 reads at one position
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def deep_bam(tmp_path_factory):
    """8000 reads at the same locus, half with 'A' at position 480 and half
    with 'G' — easily passes the >7000 read coverage filter and the 60% max
    fraction filter."""
    path = tmp_path_factory.mktemp("bam") / "deep.bam"
    reads = []
    base = list("N" * 50)
    for i in range(4000):
        qs = base.copy()
        qs[30] = "A"
        reads.append((f"da{i}", "CELL_A", LS_A, "".join(qs), 450))
    for i in range(4000):
        qs = base.copy()
        qs[30] = "G"
        reads.append((f"dg{i}", "CELL_B", LS_A, "".join(qs), 450))
    _make_bam(path, reads)
    return str(path)


def test_get_heterozygous_positions_detects_het_site(deep_bam):
    key, het_sites = sf.get_heterozygous_positions(deep_bam, INSERTION, "ins1")
    assert key == "ins1"
    # We should see a het site at position 480 with 'A' and 'G' near 50/50.
    positions = [hs[0] for hs in het_sites]
    assert 480 in positions
    site = next(hs for hs in het_sites if hs[0] == 480)
    assert {site[1], site[2]} == {"A", "G"}
    # Both fractions should be in (0.4, 0.6).
    assert 0.4 < site[3] < 0.6
    assert 0.4 < site[4] < 0.6

"""Render the polyA2 pipeline as a PDF flowchart.

Run:
    python tools/draw_flowchart.py
"""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "flowchart.pdf")


COLORS = {
    "input": "#cfe2f3",   # light blue
    "stage": "#fff2cc",   # light yellow
    "detour": "#e6e6e6",  # light gray
    "output": "#d9ead3",  # light green
}


def make_box(ax, cx, cy, w, h, text, color):
    """Draw a rounded box centred at (cx, cy)."""
    rect = mp.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="black", linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8)
    return cx, cy, w, h


def arrow(ax, src, dst):
    """Arrow from bottom-centre of src to top-centre of dst (works for diagonal)."""
    sx, sy, _, sh = src
    dx, dy, _, dh = dst
    ax.annotate(
        "", xy=(dx, dy + dh / 2), xytext=(sx, sy - sh / 2),
        arrowprops=dict(arrowstyle="->", lw=1.3, color="black"),
    )


def make_chart():
    fig, ax = plt.subplots(figsize=(11, 15))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 32)
    ax.axis("off")

    inputs = make_box(ax, 7, 30, 12, 1.5,
        "INPUTS\nBAM (-i)   |   insertions.csv (--insertions)   |   samplesheet (-s)",
        COLORS["input"])

    s1 = make_box(ax, 7, 27.5, 12, 2.0,
        "STAGE 1  —  Candidate alleles per insertion\n"
        "get_alleles_by_tagmentation_site  (parallel over insertions)\n"
        "→ res_dict   |   $prefix$_res_dict.pkl",
        COLORS["stage"])

    s2 = make_box(ax, 7, 24.7, 12, 2.0,
        "STAGE 2  —  Filter to confident allele calls\n"
        "Keep alleles with ≥10% tag-site share & insertions with ≥200 cells\n"
        "→ allele_calls   |   $prefix$_allele_calls.pkl",
        COLORS["stage"])

    s3 = make_box(ax, 7, 21.6, 12, 2.4,
        "STAGE 3  —  Correct sequencing errors, build tract-length histograms\n"
        "find_transformation_errors_per_allele  →  correction_to_apply  →  get_histogram_dfs (parallel)\n"
        "→ histogram_dfs_res_dict   |   $prefix$_histogram_dfs.pkl",
        COLORS["stage"])

    s4 = make_box(ax, 7, 18.5, 12, 2.0,
        "STAGE 4  —  Detect biallelic signal from peak position\n"
        "scan_for_biallelic_signal  (only single-allele insertions)\n"
        "→ peak_based_calling_result_dict   |   $prefix$_peak_based_calling.pkl",
        COLORS["stage"])

    het = make_box(ax, 7, 15.7, 12, 1.5,
        "Heterozygous SNP sites  —  get_heterozygous_positions\n"
        "(lazy: skipped if all of Stages 5–7 are cached)",
        COLORS["detour"])

    s5 = make_box(ax, 2.5, 12.3, 4.5, 2.4,
        "STAGE 5\nSNP CPM calling\nget_L1_flank_to_snp_cpm\n"
        "(insertions with het sites)\n→ $prefix$_snp_cpm.pkl",
        COLORS["stage"])

    s6 = make_box(ax, 7, 12.3, 4.5, 2.4,
        "STAGE 6\nPhase alleles × SNV\nphase_L1_and_flank_SNV\n"
        "(multi-allele insertions)\n→ $prefix$_phase_allele_snv.pkl",
        COLORS["stage"])

    s7 = make_box(ax, 11.5, 12.3, 4.5, 2.4,
        "STAGE 7\nPhase biallelic × SNV\nphase_biallelic_signal_and_SNV\n"
        "(biallelic insertions)\n→ $prefix$_phase_biallele_snv.pkl",
        COLORS["stage"])

    s8 = make_box(ax, 7, 8.5, 12, 2.0,
        "STAGE 8  —  Per-insertion summary\n"
        "build_insertion_summary\n"
        "→ $prefix$_insertion_summary.csv",
        COLORS["output"])

    # Arrows down the spine
    arrow(ax, inputs, s1)
    arrow(ax, s1, s2)
    arrow(ax, s2, s3)
    arrow(ax, s3, s4)
    arrow(ax, s4, het)
    # Branching to 5/6/7 from het
    arrow(ax, het, s5)
    arrow(ax, het, s6)
    arrow(ax, het, s7)
    # Merging back to s8
    arrow(ax, s5, s8)
    arrow(ax, s6, s8)
    arrow(ax, s7, s8)

    # Cache-and-resume footnote
    ax.text(
        7, 4.5,
        "Cache & resume\n"
        "Each stage checks for its output file before running. If the file exists and --overwrite is\n"
        "not set, the cached pickle is loaded and the stage is skipped (with a printed warning).",
        ha="center", va="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff4e6", edgecolor="gray"),
    )

    # Legend
    legend_items = [
        ("Input",  COLORS["input"]),
        ("Pipeline stage",  COLORS["stage"]),
        ("Conditional / intermediate",  COLORS["detour"]),
        ("Final output", COLORS["output"]),
    ]
    for i, (label, color) in enumerate(legend_items):
        x = 1 + i * 3.2
        ax.add_patch(mp.Rectangle((x, 1.4), 0.5, 0.5, facecolor=color, edgecolor="black"))
        ax.text(x + 0.7, 1.65, label, fontsize=8, va="center")

    ax.set_title("polyA2 pipeline — control & data flow", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    make_chart()

"""
General-purpose polyA mutation-rate analysis pipeline.

Wraps the core functions from clone_consistency_analysis.py with a
command-line interface that works on any dataset, not just the IVV cohort.

Required inputs
---------------
--data-dir DIR      Directory containing the pipeline pickle files.
--prefix STR        Filename prefix shared by all pickles (default: '').
                    E.g. --prefix IVV__ to load IVV__histogram_dfs.pkl etc.

Always required pickles
    {prefix}histogram_dfs.pkl
    {prefix}allele_calls.pkl

Optional pickles (skipped gracefully if absent)
    {prefix}peak_based_calling.pkl   biallelic loci via length peaks
    {prefix}phase_allele_snv.pkl     SNP phasing for sequence-based biallelic
    {prefix}phase_biallele_snv.pkl   SNP phasing for peak-based biallelic

Optional analysis modules
--------------------------
--clone-map CSV
    Two-column CSV (no header, or header 'cell_barcode,clone_group').
    Maps DNA cell barcodes to integer clone group IDs.
    Clone groups <= 0 are excluded.  If omitted, all clone-level analysis
    (consistency calibration, clone mutation calling) is skipped.

--sample-map CSV
    CSV mapping DNA cell barcodes to sample metadata.  Required columns:
        cell_barcode   str   DNA barcode
        sample         str   sample / library name
    Optional columns (enable per-sample aggregation and regression):
        time_point     int   ordinal time point (1, 2, 3, ...)
        donor          str   donor or replicate identifier
        cell_type      str   cell-type label (any string)
        is_timeseries  bool  include in time-series regression (default True)
    If omitted, per-cell outputs are saved but no per-sample summary or
    regression is produced.

Usage examples
--------------
# Minimal (no clone data, no sample metadata)
python polyA_analysis.py --data-dir /path/to/run --prefix RUN__

# With clone data
python polyA_analysis.py --data-dir /path/to/run --prefix RUN__ \\
    --clone-map /path/to/clone_map.csv

# Full analysis
python polyA_analysis.py --data-dir /path/to/run --prefix RUN__ \\
    --clone-map /path/to/clone_map.csv \\
    --sample-map /path/to/sample_map.csv \\
    --n-jobs 8
"""

import sys
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats as _stats

# ── Locate support libraries ──────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from support_functions import build_noise_model, call_mutations, add_cross_tract_consistency
from polyA_functions import (
    make_output_dir,
    compute_tested_tracts_per_cell,
    compute_epsilon_weighted_denominator,
    compute_per_cell_noise_corrected,
    add_snp_concordance,
    clone_cell_counts_from_histograms,
    annotate_clone_consistency,
    summarise_clone_consistency,
    add_clone_pvalue,
    flag_germline_alleles,
    clone_size_breakdown,
    fit_calibrated_fpr,
    fig_noise_model,
    fig_per_sample_rates,
    fig_timeseries,
    fig_allele_tiers,
    fig_cross_tract,
    fig_coverage,
    fig_pvalue_exploration,
    fig_snp_concordance,
    fig_lenadj_rate,
    fig_clone_calibration,
    fig_clone_mutations,
    fig_fpr_calibration,
)


# ── Generic data-loading helpers ─────────────────────────────────────────────

def load_pickle_optional(path):
    """Return unpickled object if path exists, else None."""
    if path and os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def load_clone_map(path):
    """
    Load a clone map CSV → dict {cell_barcode: clone_group (int)}.

    Accepts:
    - Two-column CSV with or without header (cell_barcode, clone_group).
    - Clone groups <= 0 are excluded (convention: 0 = unassigned).
    """
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, header=0)
    # Tolerate headerless files: if first column looks like a barcode, use as-is
    if df.columns[0].isdigit() or len(df.columns[0]) > 20:
        df.columns = ['cell_barcode', 'clone_group']
    else:
        df = df.rename(columns={df.columns[0]: 'cell_barcode',
                                  df.columns[1]: 'clone_group'})
    df['clone_group'] = pd.to_numeric(df['clone_group'], errors='coerce')
    df = df[df['clone_group'].notna() & (df['clone_group'] > 0)]
    return df.set_index('cell_barcode')['clone_group'].astype(int).to_dict()


def load_sample_map(path):
    """
    Load a sample map CSV → pd.DataFrame indexed by cell_barcode.

    Required columns : cell_barcode, sample
    Optional columns : time_point, donor, cell_type, is_timeseries
    """
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'cell_barcode' not in df.columns or 'sample' not in df.columns:
        raise ValueError("--sample-map must have columns: cell_barcode, sample")
    for col, default in [('time_point', np.nan), ('donor', None),
                          ('cell_type', None), ('is_timeseries', True)]:
        if col not in df.columns:
            df[col] = default
    df['time_point'] = pd.to_numeric(df['time_point'], errors='coerce')
    df['is_timeseries'] = df['is_timeseries'].astype(bool)
    return df.set_index('cell_barcode')


def load_rna_clone_map(clone_path, samplesheet_path):
    """
    Load a clone file with RNA barcodes and map them to DNA barcodes.

    Expected columns in clone_path:
        Cell_ID    str   RNA barcode (e.g. FL1_AACGGACTCGGTTCTAACTC)
        Clone_ID   str   Clone identifier (mutation string or integer label)
        celltype   str   Cell-type label

    samplesheet_path: CSV with columns including Modality, Plate_code, WellID
        and barcodes as the index.  DNA and RNA cells at the same
        (Plate_code, WellID) are treated as the same physical cell.

    Returns
    -------
    clone_map  : dict {dna_barcode: clone_group_int}
    sample_map : pd.DataFrame indexed by dna_barcode with columns
                 sample, cell_type, time_point, donor, is_timeseries
                 (sample = celltype, time_point = NaN, is_timeseries = False)
    clone_id_legend : dict {clone_group_int: original_Clone_ID_str}
    """
    cf = pd.read_csv(clone_path)
    # Drop leading unnamed index column (produced when a DataFrame is saved with default index)
    if cf.columns[0].startswith('Unnamed') or cf.columns[0] == '':
        cf = cf.iloc[:, 1:]
    # Deduplicate column names (e.g. Cell_ID appearing twice), keeping first occurrence
    cf = cf.loc[:, ~cf.columns.duplicated(keep='first')]
    required = {'Cell_ID', 'Clone_ID', 'celltype'}
    if not required.issubset(cf.columns):
        raise ValueError(f"--rna-clone-map must have columns: {required}")

    ss = pd.read_csv(samplesheet_path, index_col=0)
    # Drop duplicate first column (some samplesheets have the barcode column twice)
    ss = ss.loc[:, ~ss.columns.duplicated(keep='first')]
    if not {'Modality', 'Plate_code', 'WellID'}.issubset(ss.columns):
        raise ValueError("--samplesheet must have columns: Modality, Plate_code, WellID")

    def _reset_bc(df, new_name):
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: new_name})
        return df

    dna = _reset_bc(ss[ss['Modality'] == 'DNA'][['Plate_code', 'WellID']], 'dna_bc')
    rna = _reset_bc(ss[ss['Modality'] == 'RNA'][['Plate_code', 'WellID']], 'rna_bc')
    rna_to_dna = (dna.merge(rna, on=['Plate_code', 'WellID'])
                  .set_index('rna_bc')['dna_bc'].to_dict())

    cell_ids = cf['Cell_ID']
    # If samplesheet barcodes don't carry the sample prefix (e.g. 'FL3_'), strip it
    if not cell_ids.isin(rna_to_dna).any():
        cell_ids = cell_ids.str.replace(r'^[^_]+_', '', regex=True)
    cf['dna_bc'] = cell_ids.map(rna_to_dna)
    matched = cf.dropna(subset=['dna_bc'])

    # Encode Clone_ID strings → stable integers
    unique_clones = sorted(matched['Clone_ID'].astype(str).unique())
    clone_id_to_int = {c: i + 1 for i, c in enumerate(unique_clones)}
    clone_id_legend = {v: k for k, v in clone_id_to_int.items()}

    clone_map = (matched[['dna_bc', 'Clone_ID']]
                 .assign(clone_group=matched['Clone_ID'].astype(str).map(clone_id_to_int))
                 .set_index('dna_bc')['clone_group'].astype(int).to_dict())

    # sample_map covers ALL DNA cells in the samplesheet (not just those in the clone file)
    # Use the samplesheet Sample column as the grouping
    if 'Sample' not in ss.columns:
        raise ValueError("--samplesheet must have a 'Sample' column for per-sample grouping")
    dna_all = _reset_bc(ss[ss['Modality'] == 'DNA'][['Sample']], 'cell_barcode')
    donor_col = ss['Donor'] if 'Donor' in ss.columns else None
    if donor_col is not None:
        dna_all = _reset_bc(ss[ss['Modality'] == 'DNA'][['Sample', 'Donor']], 'cell_barcode')
        sample_map = (dna_all.rename(columns={'Sample': 'sample', 'Donor': 'donor'})
                      .assign(cell_type=dna_all['Sample'].values,
                              time_point=np.nan, is_timeseries=False)
                      .set_index('cell_barcode'))
    else:
        sample_map = (dna_all.rename(columns={'Sample': 'sample'})
                      .assign(cell_type=dna_all['Sample'].values,
                              time_point=np.nan, donor=None, is_timeseries=False)
                      .set_index('cell_barcode'))

    return clone_map, sample_map, clone_id_legend


# ── Per-sample aggregation (general version) ──────────────────────────────────

def summarise_by_sample_general(per_cell, sample_map):
    """
    Aggregate per-cell counts by sample using a generic sample_map DataFrame.

    Returns pd.DataFrame with one row per sample.
    """
    df = per_cell.copy()
    df = df.join(sample_map[['sample', 'time_point', 'donor',
                               'cell_type', 'is_timeseries']],
                  on='cell', how='inner')

    def _se(x):
        return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan

    agg_spec = dict(
        n_cells=('cell', 'count'),
        is_timeseries=('is_timeseries', 'first'),
        donor=('donor', 'first'),
        time_point=('time_point', 'first'),
        cell_type=('cell_type', 'first'),
        mean_corrected_total=('n_corrected_total', 'mean'),
        se_corrected_total=('n_corrected_total', _se),
        mean_corrected_pm1=('n_corrected_pm1', 'mean'),
        se_corrected_pm1=('n_corrected_pm1', _se),
        mean_corrected_pm2=('n_corrected_pm2', 'mean'),
        se_corrected_pm2=('n_corrected_pm2', _se),
        mean_corrected_ge3=('n_corrected_ge3', 'mean'),
        se_corrected_ge3=('n_corrected_ge3', _se),
        mean_is_mutated=('n_is_mutated', 'mean'),
        mean_n_tested=('n_tested_tracts', 'mean'),
        mean_mutation_rate=('mutation_rate', 'mean'),
        se_mutation_rate=('mutation_rate', _se),
        mean_clean_single_rate=('clean_single_rate', 'mean'),
        se_clean_single_rate=('clean_single_rate', _se),
        mean_single_obs_cands=('n_cand_single', 'mean'),
        mean_corrected_clean_single=('n_corrected_clean_single', 'mean'),
        se_corrected_clean_single=('n_corrected_clean_single', _se),
    )
    if 'mutation_rate_lenadj' in df.columns:
        agg_spec['mean_mutation_rate_lenadj'] = ('mutation_rate_lenadj', 'mean')
        agg_spec['se_mutation_rate_lenadj']   = ('mutation_rate_lenadj', _se)
        agg_spec['mean_median_epsilon']        = ('median_epsilon', 'mean')

    return (df.groupby('sample', sort=False)
              .agg(**agg_spec)
              .reset_index()
              .sort_values(['is_timeseries', 'donor', 'time_point'], na_position='last'))


# ── Regression helper ─────────────────────────────────────────────────────────

def run_regression(ts, metrics, n_clust=None, ts_cells=None, noise_fn_name='OLS'):
    """
    Run pooled OLS regressions on per-sample time-series data.

    ts      : DataFrame filtered to time-series samples, must have time_point column
    metrics : list of (column, label) pairs
    Returns list of regression row dicts compatible with regression_results.csv.
    """
    import statsmodels.api as sm

    rows = []
    x = ts['time_point'].values.astype(float)

    for col, label in metrics:
        if col not in ts.columns:
            continue
        y = ts[col].values
        slope, intercept, r, p_val, _ = _stats.linregress(x, y)
        p1s = p_val / 2 if slope > 0 else 1.0
        for sided, is_1s in [('two-sided', False), ('one-sided', True)]:
            rows.append(dict(label=f'{label} — pooled OLS {sided}',
                             slope=slope, r2=r**2,
                             p_two_sided=p_val, p_one_sided=p1s,
                             is_one_sided=is_1s, n=len(ts)))

        # Cell-level cluster-robust OLS if cell data provided
        if ts_cells is not None and col.replace('mean_', '') in ts_cells.columns:
            cell_col = col.replace('mean_', '')
            X = sm.add_constant(ts_cells['time_point'].values)
            y_c = ts_cells[cell_col].values
            fit = sm.OLS(y_c, X).fit()
            rob = fit.get_robustcov_results(
                cov_type='cluster', groups=ts_cells['sample'].values)
            coef, p_rob = rob.params[1], rob.pvalues[1]
            p_rob1s = p_rob / 2 if coef > 0 else 1.0
            for sided, is_1s in [('two-sided', False), ('one-sided', True)]:
                rows.append(dict(
                    label=f'{label} — cell cluster-robust OLS {sided}',
                    slope=coef, r2=np.nan,
                    p_two_sided=p_rob, p_one_sided=p_rob1s,
                    is_one_sided=is_1s,
                    n=ts_cells['sample'].nunique()))
    return rows


# ── Pre-analysis insertion filter ─────────────────────────────────────────────

def build_insertion_blacklist(insertion_info_path, canon_chroms_re):
    ins = pd.read_csv(insertion_info_path, index_col=0)
    blacklist = set()
    if 'chrom' in ins.columns:
        bad_chrom = ins[~ins['chrom'].astype(str).str.match(canon_chroms_re)].index
        blacklist.update(bad_chrom.tolist())
    return blacklist


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='polyA mutation-rate analysis (general dataset)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--data-dir', required=True,
                        help='Directory containing pickle files')
    parser.add_argument('--prefix', default='',
                        help='Filename prefix for pickles (default: "")')
    parser.add_argument('--clone-map', default=None,
                        help='CSV: cell_barcode, clone_group  (optional)')
    parser.add_argument('--rna-clone-map', default=None,
                        help='CSV with Cell_ID (RNA barcode), Clone_ID, celltype. '
                             'Requires --samplesheet for RNA→DNA barcode matching.')
    parser.add_argument('--samplesheet', default=None,
                        help='Samplesheet CSV (index=barcode, columns include '
                             'Modality, Plate_code, WellID). Required with --rna-clone-map.')
    parser.add_argument('--sample-map', default=None,
                        help='CSV: cell_barcode, sample [, time_point, donor, ...] (optional)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: auto-timestamped)')
    parser.add_argument('--insertion-info', default=None,
                        help='Insertion summary CSV (index=ins_key, column chrom required). '
                             'When provided, insertions on non-canonical chromosomes '
                             '(unplaced scaffolds, alt contigs) are excluded before analysis.')
    parser.add_argument('--max-pop-af', type=float, default=0.005,
                        help='Max population allele frequency for clone-consistent mutations '
                             '(default: 0.005 = 0.5%%). After calling mutations, any '
                             'clone-consistent call where the mutant length is present in '
                             'more than this fraction of ALL cells is flagged as likely '
                             'germline and excluded.')
    parser.add_argument('--max-seq-minor-af', type=float, default=0.05,
                        help='Max sequence minor-allele frequency for unphased clone '
                             'mutations (default: 0.05). Requires --insertion-info. '
                             'Unphased calls (a_key=0) at loci where a second sequence '
                             'allele exceeds this population frequency are flagged as '
                             'potentially germline.')
    parser.add_argument('--canon-chroms', default=r'^chr[0-9XYM]+$',
                        help='Regex matching canonical chromosome names to KEEP '
                             '(default: "^chr[0-9XYM]+$"). Insertions on non-matching '
                             'chromosomes (unplaced scaffolds, alt contigs) are excluded.')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel workers (default: 1)')
    parser.add_argument('--min-tested', type=int, default=400,
                        help='Min tested tracts per cell (default: 400)')
    args = parser.parse_args()

    n_jobs    = args.n_jobs
    MIN_TESTED = args.min_tested
    data_dir  = args.data_dir
    prefix    = args.prefix
    if args.output_dir:
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
    else:
        out_dir = make_output_dir(explicit=None)
    print(f'Output directory: {out_dir}')

    def _pkl(name):
        return os.path.join(data_dir, f'{prefix}{name}.pkl')

    # ── Load required data ────────────────────────────────────────────────────
    print('Loading data ...')
    for req in ('histogram_dfs', 'allele_calls'):
        if not os.path.exists(_pkl(req)):
            sys.exit(f'ERROR: required pickle not found: {_pkl(req)}')

    with open(_pkl('histogram_dfs'), 'rb') as f:
        histogram_dfs = pickle.load(f)
    with open(_pkl('allele_calls'), 'rb') as f:
        allele_calls = pickle.load(f)

    peak_based         = load_pickle_optional(_pkl('peak_based_calling'))
    phase_allele_snv   = load_pickle_optional(_pkl('phase_allele_snv'))
    phase_biallele_snv = load_pickle_optional(_pkl('phase_biallele_snv'))

    have_snp = phase_allele_snv is not None or phase_biallele_snv is not None
    print(f'  peak_based_calling : {"loaded" if peak_based         else "not found — skipping biallelic peaks"}')
    print(f'  phase_allele_snv   : {"loaded" if phase_allele_snv   else "not found"}')
    print(f'  phase_biallele_snv : {"loaded" if phase_biallele_snv else "not found"}')

    # ── Load optional clone / sample maps ─────────────────────────────────────
    clone_id_legend = {}
    if args.rna_clone_map:
        if not args.samplesheet:
            sys.exit('ERROR: --rna-clone-map requires --samplesheet for RNA→DNA barcode matching')
        dna_to_clone, rna_sample_map, clone_id_legend = load_rna_clone_map(
            args.rna_clone_map, args.samplesheet)
        # Use the derived sample map unless an explicit one was provided
        sample_map = load_sample_map(args.sample_map) if args.sample_map else rna_sample_map
        sample_map.index.name = 'cell_barcode'
    else:
        dna_to_clone = load_clone_map(args.clone_map)
        sample_map   = load_sample_map(args.sample_map)

    have_clone   = dna_to_clone is not None and len(dna_to_clone) > 0
    have_sample  = sample_map is not None and len(sample_map) > 0
    print(f'  clone map          : {"loaded ({:,} cells)".format(len(dna_to_clone)) if have_clone else "not provided — clone analysis skipped"}')
    print(f'  sample map         : {"loaded ({:,} cells)".format(len(sample_map))   if have_sample else "not provided — per-sample analysis skipped"}')

    # ── Pre-analysis insertion filter ─────────────────────────────────────────
    insertion_info_df = None
    if args.insertion_info:
        insertion_info_df = pd.read_csv(args.insertion_info, index_col=0)
        blacklist = build_insertion_blacklist(args.insertion_info, args.canon_chroms)
        n_total = len(histogram_dfs)
        for d in [histogram_dfs, allele_calls,
                  peak_based, phase_allele_snv, phase_biallele_snv]:
            if d is not None:
                for k in blacklist:
                    d.pop(k, None)
        print(f'  Pre-filter: excluded {len(blacklist)} of {n_total} insertions '
              f'(non-canonical chromosome)')

    # ── Core pipeline ─────────────────────────────────────────────────────────
    print(f'\nBuilding noise model (n_jobs={n_jobs}) ...')
    noise_df, noise_fn = build_noise_model(
        histogram_dfs, peak_based_calling=peak_based or {}, n_jobs=n_jobs)

    print(f'Running mutation caller (n_jobs={n_jobs}) ...')
    mutations, tested_cells = call_mutations(
        histogram_dfs, allele_calls, noise_fn,
        peak_based_calling=peak_based or {},
        phase_allele_snv_dict=phase_allele_snv or {},
        min_tag_sites=1, p_threshold=0.05,
        n_jobs=n_jobs,
    )

    print(f'Adding cross-tract consistency (n_jobs={n_jobs}) ...')
    mutations = add_cross_tract_consistency(
        mutations, histogram_dfs,
        peak_based_calling=peak_based or {}, n_jobs=n_jobs)

    if have_snp:
        print('Annotating SNP concordance ...')
        mutations = add_snp_concordance(
            mutations,
            phase_allele_snv   or {},
            phase_biallele_snv or {})

    # ── Per-cell denominators ─────────────────────────────────────────────────
    print(f'Computing tested-tract denominator (n_jobs={n_jobs}) ...')
    tested_tracts = compute_tested_tracts_per_cell(
        histogram_dfs, peak_based_calling=peak_based or {}, n_jobs=n_jobs)

    print(f'Computing ε(L)-weighted denominator (n_jobs={n_jobs}) ...')
    eps_denom = compute_epsilon_weighted_denominator(
        histogram_dfs, noise_fn, peak_based_calling=peak_based or {}, n_jobs=n_jobs)

    # ── Clone analysis (optional) ─────────────────────────────────────────────
    annotated      = None
    clone_mut      = None
    summary        = None
    breakdown      = {}
    calibrated_fpr = {}

    if have_clone:
        print('Counting clone cells per tract ...')
        ccc = clone_cell_counts_from_histograms(
            histogram_dfs, dna_to_clone, n_jobs=n_jobs, min_tag_sites=1)

        print('Annotating with clone consistency ...')
        annotated = annotate_clone_consistency(mutations, ccc, dna_to_clone)

        print('Summarising clone consistency ...')
        summary, clone_mut = summarise_clone_consistency(annotated, min_clone_size=2)
        clone_mut = add_clone_pvalue(clone_mut, annotated, tested_cells)
        clone_mut = flag_germline_alleles(
            clone_mut, histogram_dfs,
            max_pop_af=args.max_pop_af,
            insertion_info=insertion_info_df,
            max_seq_minor_af=args.max_seq_minor_af,
        )

        print('\n=== Clone consistency summary ===')
        print(summary.to_string())

        strong = (clone_mut[clone_mut['is_clone_mutated']].sort_values('p_fisher')
                  if 'is_clone_mutated' in clone_mut.columns else clone_mut.iloc[0:0])
        print(f'\n=== Clone-consistent mutations: {len(strong)} ===')
        if len(strong):
            show_cols = [c for c in ['ins_key','a_key','t_key','clone_group',
                                      'ref_length','cell_length','delta_length',
                                      'n_clone_cells','n_clone_calling',
                                      'confidence_tier','q_fisher',
                                      'snp_concordance','snp_concordance_low']
                         if c in strong.columns]
            print(strong[show_cols].head(20).to_string())

        breakdown      = clone_size_breakdown(annotated)
        calibrated_fpr = fit_calibrated_fpr(breakdown)
        print('\n=== Calibrated per-cell FPR ===')
        for bucket, fpr in calibrated_fpr.items():
            print(f'  {bucket}: {fpr:.4f}')
    else:
        annotated = mutations  # no clone columns; downstream functions degrade gracefully

    # ── Per-cell noise-corrected counts ───────────────────────────────────────
    print('\n=== Per-cell noise-corrected mutation counts ===')
    per_cell = compute_per_cell_noise_corrected(
        annotated, calibrated_fpr or None,
        single_obs_epsilon_threshold=0.05)

    per_cell = per_cell.merge(
        tested_tracts.rename('n_tested_tracts').reset_index()
                     .rename(columns={'index': 'cell'}),
        on='cell', how='left')
    per_cell = per_cell.merge(eps_denom.reset_index(), on='cell', how='left')

    per_cell['mutation_rate'] = (per_cell['n_corrected_total']
                                  / per_cell['n_tested_tracts'].replace(0, np.nan))
    per_cell['clean_single_rate'] = (per_cell['n_corrected_clean_single']
                                      / per_cell['n_tested_tracts'].replace(0, np.nan))
    per_cell['mutation_rate_lenadj'] = (per_cell['n_corrected_total']
                                         / per_cell['epsilon_weighted_denom'].replace(0, np.nan))

    per_cell_all = per_cell.copy()
    per_cell = per_cell[per_cell['n_tested_tracts'] >= MIN_TESTED].copy()
    print(f'Cells with ≥1 candidate:          {len(per_cell_all):,}')
    print(f'After ≥{MIN_TESTED} tested filter: {len(per_cell):,}')
    print(per_cell['mutation_rate'].describe(percentiles=[.25,.5,.75,.9,.95]).to_string())

    # ── Per-sample summary (optional) ─────────────────────────────────────────
    sample_summary = pd.DataFrame()
    ts             = pd.DataFrame()
    tp_stats       = pd.DataFrame()
    ts_cells       = pd.DataFrame()
    regression_rows = []

    if have_sample:
        print('\n=== Per-sample summary ===')
        per_cell['cell'] = per_cell['cell'].astype(str)
        sample_summary = summarise_by_sample_general(per_cell, sample_map)
        per_cell = per_cell.join(sample_map[['sample']], on='cell', how='left')
        print(sample_summary[['sample','n_cells','mean_mutation_rate',
                               'se_mutation_rate']].to_string(index=False))

        ts = sample_summary[sample_summary['is_timeseries'] &
                             sample_summary['time_point'].notna()].copy()
        ts['time_point'] = ts['time_point'].astype(float)

        if len(ts) >= 3:
            print('\n=== Time-series regression ===')

            # Build cell-level table for cluster-robust OLS
            all_cells = (tested_tracts[tested_tracts >= MIN_TESTED]
                         .rename('n_tested_tracts').reset_index()
                         .rename(columns={'index': 'cell'}))
            for rate_col in ('mutation_rate', 'mutation_rate_lenadj'):
                if rate_col in per_cell.columns:
                    all_cells = all_cells.merge(
                        per_cell[['cell', rate_col]].assign(**{
                            rate_col: per_cell[rate_col].fillna(0)}),
                        on='cell', how='left')
                    all_cells[rate_col] = all_cells[rate_col].fillna(0)
            all_cells['cell'] = all_cells['cell'].astype(str)
            all_cells = all_cells.join(
                sample_map[['sample', 'time_point']], on='cell', how='inner')
            all_cells = all_cells[all_cells['time_point'].notna()].copy()
            all_cells['time_point'] = all_cells['time_point'].astype(float)
            ts_cells = all_cells.copy()

            metrics = [
                ('mean_mutation_rate',        'All-tract mutation rate'),
                ('mean_clean_single_rate',    'Clean-tract rate (ε≤0.05)'),
                ('mean_mutation_rate_lenadj', 'Length-adjusted mutation rate'),
            ]
            regression_rows = run_regression(ts, metrics, ts_cells=ts_cells)

            for row in regression_rows:
                if not row['is_one_sided']:
                    print(f'  {row["label"]}: slope={row["slope"]:.5f}, '
                          f'R²={row["r2"] if pd.notna(row["r2"]) else "n/a":.3f}, '
                          f'p(1s)={row["p_one_sided"]:.4f}')

            tp_stats = (ts.groupby('time_point')['mean_mutation_rate']
                        .agg(tp_mean='mean', tp_std='std', n_samples='count')
                        .reset_index())
            tp_stats['tp_se'] = tp_stats['tp_std'] / np.sqrt(tp_stats['n_samples'])

    # ── Save results ──────────────────────────────────────────────────────────
    print(f'\nSaving results to {out_dir} ...')
    tbl_dir = os.path.join(out_dir, 'tables')
    os.makedirs(tbl_dir, exist_ok=True)
    noise_df.to_csv(os.path.join(tbl_dir, 'noise_model.csv'), index=False)
    per_cell_all.to_csv(os.path.join(tbl_dir, 'per_cell.csv'), index=False)
    if not sample_summary.empty:
        sample_summary.to_csv(os.path.join(tbl_dir, 'per_sample_summary.csv'), index=False)
    if not ts.empty:
        ts.to_csv(os.path.join(tbl_dir, 'timeseries_samples.csv'), index=False)
        tp_stats.to_csv(os.path.join(tbl_dir, 'timeseries_tp_stats.csv'), index=False)
    if regression_rows:
        pd.DataFrame(regression_rows).to_csv(
            os.path.join(tbl_dir, 'regression_results.csv'), index=False)
    if have_clone and clone_mut is not None:
        clone_mut.to_csv(os.path.join(tbl_dir, 'clone_consistent_mutations.csv'), index=False)
        clone_mut[clone_mut.get('is_clone_mutated', pd.Series(False, index=clone_mut.index))].to_csv(
            os.path.join(tbl_dir, 'clone_consistent_mutations_filtered.csv'), index=False)
        if summary is not None:
            summary.to_csv(os.path.join(tbl_dir, 'clone_consistency_summary.csv'))
        if calibrated_fpr:
            pd.DataFrame([calibrated_fpr]).to_csv(
                os.path.join(tbl_dir, 'calibrated_fpr.csv'), index=False)
    if clone_id_legend:
        (pd.DataFrame({'clone_group': list(clone_id_legend.keys()),
                       'Clone_ID':    list(clone_id_legend.values())})
         .to_csv(os.path.join(tbl_dir, 'clone_id_legend.csv'), index=False))

    # ── Figures ───────────────────────────────────────────────────────────────
    print('  Generating figures ...')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    def _try(name, fn):
        try:
            fn()
            print(f'    {name}.pdf  ✓')
        except Exception as exc:
            print(f'    {name}.pdf  FAILED: {exc}')

    _try('fig1_noise_model',
         lambda: fig_noise_model(noise_df, out_dir))

    if have_clone and breakdown:
        _try('fig2_clone_calibration',
             lambda: fig_clone_calibration(breakdown, calibrated_fpr, out_dir))

    if have_sample and not sample_summary.empty:
        _try('fig3_per_sample_rates',
             lambda: fig_per_sample_rates(sample_summary, per_cell, out_dir))
        if not ts.empty:
            _try('fig4_timeseries',
                 lambda: fig_timeseries(ts, tp_stats, ts_cells, out_dir))
            _try('ed7_lenadj_rate',
                 lambda: fig_lenadj_rate(ts, sample_summary, out_dir))

    if have_clone and clone_mut is not None:
        _try('fig5_clone_mutations',
             lambda: fig_clone_mutations(clone_mut, out_dir))

    _try('ed1_allele_tiers',
         lambda: fig_allele_tiers(mutations, allele_calls, out_dir))
    _try('ed2_cross_tract',
         lambda: fig_cross_tract(annotated, out_dir))

    if have_sample and not sample_summary.empty:
        _try('ed3_coverage',
             lambda: fig_coverage(per_cell, sample_summary, MIN_TESTED, out_dir))

    if have_clone and breakdown:
        _try('ed4_fpr_calibration',
             lambda: fig_fpr_calibration(breakdown, calibrated_fpr, summary, out_dir))

    if regression_rows:
        _try('ed5_pvalue_exploration',
             lambda: fig_pvalue_exploration(regression_rows, out_dir))

    if have_snp:
        _try('ed6_snp_concordance',
             lambda: fig_snp_concordance(
                 phase_allele_snv or {}, phase_biallele_snv or {},
                 allele_calls, out_dir))

    print(f'\nDone.  All output in: {out_dir}')


if __name__ == '__main__':
    main()

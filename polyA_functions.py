"""
General-purpose polyA analysis functions.

Contains the core denominators, noise-corrected counts, SNP concordance,
and figure-generation routines used by polyA_analysis.py.  These are
extracted from clone_consistency_analysis.py so that the general-purpose
pipeline does not require that file on the path.

Clone-specific functions (annotate_clone_consistency, fit_calibrated_fpr,
etc.) remain in clone_consistency_analysis.py.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.size']    = 7
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

TP_COLORS     = {1: '#2166ac', 2: '#f4a582', 3: '#d6604d'}
DONOR_COLORS  = {'D2': '#1f77b4', 'D3': '#ff7f0e'}
TIER_COLORS   = {1: '#d62728', 2: '#2ca02c', 3: '#ff7f0e', 4: '#9467bd', 5: '#7f7f7f'}
BUCKET_COLORS = {'±1': '#e41a1c', '±2': '#377eb8', '|Δ|≥3': '#4daf4a'}


# ── Output directory ──────────────────────────────────────────────────────────

def make_output_dir(base='results', explicit=None):
    """Create timestamped output directory tree and return its path."""
    if explicit:
        root = explicit
    else:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            base, datetime.now().strftime('%Y-%m-%d_%H-%M'))
    for sub in ('figures', 'tables', 'pickles'):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


# ── Tested-tract denominator ──────────────────────────────────────────────────

def _tested_tracts_one_ins(ins_key, df, value_key, min_tag_sites, biallele_result):
    from support_functions import _assign_biallelic_a_key
    df_r = df.reset_index()
    if value_key not in df_r.columns or df_r[value_key].isna().all():
        return {}
    if biallele_result is not None:
        df_split = _assign_biallelic_a_key(df_r, biallele_result)
        if df_split is not None:
            df_r = df_split
    df_r = df_r[df_r[value_key].notna()]
    if df_r.empty:
        return {}
    obs = (df_r.groupby(['cell', 'a_key', 't_key'])['tag_site']
           .count()
           .reset_index(name='n_obs'))
    tested = obs[obs['n_obs'] >= min_tag_sites]
    return tested.groupby('cell')['a_key'].count().to_dict()


def compute_tested_tracts_per_cell(histogram_dfs, peak_based_calling=None,
                                    value_key='hist_peak', min_tag_sites=1,
                                    n_jobs=1):
    """
    For each cell, count (insertion, allele, tract) combinations with
    >= min_tag_sites valid observations.  Denominator for mutation rates.
    Returns pd.Series {cell: n_tested_tracts}.
    """
    from joblib import Parallel, delayed as _delayed

    if peak_based_calling is None:
        peak_based_calling = {}
    results = Parallel(n_jobs=n_jobs)(
        _delayed(_tested_tracts_one_ins)(
            ins_key, df, value_key, min_tag_sites,
            peak_based_calling.get(ins_key),
        )
        for ins_key, df in histogram_dfs.items()
    )
    combined = {}
    for partial in results:
        for cell, n in partial.items():
            combined[cell] = combined.get(cell, 0) + n
    return pd.Series(combined, name='n_tested_tracts')


# ── ε(L)-weighted denominator ─────────────────────────────────────────────────

def _epsilon_denom_one_ins(ins_key, df, noise_fn, value_key, min_tag_sites, biallele_result):
    from support_functions import _assign_biallelic_a_key, _reference_lengths
    df_r = df.reset_index()
    if value_key not in df_r.columns or df_r[value_key].isna().all():
        return {}
    if biallele_result is not None:
        df_split = _assign_biallelic_a_key(df_r, biallele_result)
        if df_split is not None:
            df_r = df_split
    df_r = df_r[df_r[value_key].notna()]
    if df_r.empty:
        return {}
    ref_lengths = _reference_lengths(df_r, value_key)
    obs = (df_r.groupby(['cell', 'a_key', 't_key'])['tag_site']
               .count()
               .reset_index(name='n_obs'))
    tested = obs[obs['n_obs'] >= min_tag_sites]
    result = {}
    for row in tested.itertuples(index=False):
        ref_L = ref_lengths.get((row.a_key, row.t_key))
        if ref_L is None:
            continue
        eps = float(np.clip(noise_fn(ref_L), 1e-9, 1.0))
        result.setdefault(row.cell, []).append(eps)
    return result


def compute_epsilon_weighted_denominator(histogram_dfs, noise_fn,
                                          peak_based_calling=None,
                                          value_key='hist_peak',
                                          min_tag_sites=1, n_jobs=1):
    """
    Per-cell ε(L)-weighted denominator for the length-adjusted mutation rate:

        ε_weighted_denom = Σ_i ε(ref_length_i) / median_cell(ε(ref_length_i))

    Returns pd.DataFrame indexed by cell with columns:
        sum_epsilon, median_epsilon, epsilon_weighted_denom
    """
    from joblib import Parallel, delayed as _delayed
    if peak_based_calling is None:
        peak_based_calling = {}
    results = Parallel(n_jobs=n_jobs)(
        _delayed(_epsilon_denom_one_ins)(
            ins_key, df, noise_fn, value_key, min_tag_sites,
            peak_based_calling.get(ins_key),
        )
        for ins_key, df in histogram_dfs.items()
    )
    combined = {}
    for partial in results:
        for cell, eps_list in partial.items():
            combined.setdefault(cell, []).extend(eps_list)
    rows = []
    for cell, eps_list in combined.items():
        arr = np.array(eps_list)
        med = float(np.median(arr))
        rows.append({
            'cell': cell,
            'sum_epsilon': float(arr.sum()),
            'median_epsilon': med,
            'epsilon_weighted_denom': float(arr.sum() / med) if med > 0 else np.nan,
        })
    return (pd.DataFrame(rows).set_index('cell')
            if rows else pd.DataFrame(columns=['sum_epsilon', 'median_epsilon',
                                                'epsilon_weighted_denom']))


# ── Per-cell noise-corrected counts ───────────────────────────────────────────

def compute_per_cell_noise_corrected(annotated, calibrated_fpr=None,
                                      single_obs_epsilon_threshold=None):
    """
    For every cell compute the noise-corrected mutation count.

    annotated       : output of call_mutations (or annotate_clone_consistency)
    calibrated_fpr  : optional dict {bucket: p_noise} from clone calibration
    """
    df = annotated.reset_index().copy()

    if calibrated_fpr is not None:
        def _bucket(d):
            return '±1' if abs(d) == 1 else ('±2' if abs(d) == 2 else '|Δ|≥3')
        df['_cal_fpr'] = df['delta_length'].apply(
            lambda d: calibrated_fpr.get(_bucket(d), 0.05))
        df['effective_fpr'] = np.where(
            df['n_tag_sites'] >= 2,
            np.maximum(df['background_rate'], df['_cal_fpr']),
            df['background_rate'],
        )
    else:
        df['effective_fpr'] = df['background_rate']

    df['_contrib']      = 1.0 - df['effective_fpr']
    df['_is_pm1']       = df['delta_length'].abs() == 1
    df['_is_pm2']       = df['delta_length'].abs() == 2
    df['_is_ge3']       = df['delta_length'].abs() >= 3
    df['_single']       = df['n_tag_sites'] == 1
    df['_clean_single'] = (df['_single'] &
                            (df['epsilon'] <= (single_obs_epsilon_threshold or np.inf)))

    result = (df.groupby('cell')
              .agg(
                  n_cand_pm1=('_is_pm1', 'sum'),
                  n_cand_pm2=('_is_pm2', 'sum'),
                  n_cand_ge3=('_is_ge3', 'sum'),
                  n_cand_single=('_single', 'sum'),
                  n_cand_clean_single=('_clean_single', 'sum'),
                  n_corrected_pm1=('_contrib',
                                   lambda x: x[df.loc[x.index, '_is_pm1']].sum()),
                  n_corrected_pm2=('_contrib',
                                   lambda x: x[df.loc[x.index, '_is_pm2']].sum()),
                  n_corrected_ge3=('_contrib',
                                   lambda x: x[df.loc[x.index, '_is_ge3']].sum()),
                  n_corrected_clean_single=('_contrib',
                                            lambda x: x[df.loc[x.index, '_clean_single']].sum()),
                  n_is_mutated=('is_mutated', 'sum'),
              )
              .reset_index())
    result['n_corrected_total'] = (result['n_corrected_pm1']
                                   + result['n_corrected_pm2']
                                   + result['n_corrected_ge3'])
    return result


# ── SNP concordance ───────────────────────────────────────────────────────────

def add_snp_concordance(mutations, phase_allele_snv, phase_biallele_snv,
                        low_concordance_threshold=0.80):
    """
    Attach snp_concordance, snp_concordance_p, snp_concordance_low columns.
    concordance = max(a00+a11, a01+a10) / total at best het site.
    """
    def _best_concordance(phase_dict):
        rows = []
        for ins_key, sites in phase_dict.items():
            best_chi2 = -1
            best_conc = np.nan
            best_p    = np.nan
            for entry in sites.values():
                allele_to_snp, chi2, p, dof, _ = entry
                if chi2 is None or p is None:
                    continue
                alleles = list(allele_to_snp.keys())
                if len(alleles) != 2:
                    continue
                c0, c1 = allele_to_snp[alleles[0]], allele_to_snp[alleles[1]]
                bases = sorted(set(list(c0.keys()) + list(c1.keys())))
                if len(bases) < 2:
                    continue
                b0, b1 = bases[0], bases[1]
                a00 = c0.get(b0, 0); a01 = c0.get(b1, 0)
                a10 = c1.get(b0, 0); a11 = c1.get(b1, 0)
                total = a00 + a01 + a10 + a11
                if total == 0:
                    continue
                conc = max(a00 + a11, a01 + a10) / total
                if chi2 > best_chi2:
                    best_chi2 = chi2
                    best_conc = conc
                    best_p    = p
            if best_chi2 >= 0:
                rows.append({'ins_key': ins_key,
                             'snp_concordance': best_conc,
                             'snp_concordance_p': best_p})
        return pd.DataFrame(rows).set_index('ins_key') if rows else pd.DataFrame()

    seq_conc = _best_concordance(phase_allele_snv)
    pbc_conc = _best_concordance(phase_biallele_snv)
    conc_df  = pd.concat([seq_conc, pbc_conc])

    m = mutations.reset_index()
    m = m.merge(conc_df.reset_index(), on='ins_key', how='left')
    m['snp_concordance_low'] = (
        m['snp_concordance_p'].lt(0.05) &
        m['snp_concordance'].lt(low_concordance_threshold)
    )
    return m.set_index(['ins_key', 'cell', 'a_key', 't_key'])


# ── Figure helpers ────────────────────────────────────────────────────────────

def _savefig(fig, path):
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def _add_caption(fig, legend, conclusion, y=-0.04, fontsize=5.5):
    text = f"Figure legend — {legend}\n\nConclusion — {conclusion}"
    fig.text(0.0, y, text, fontsize=fontsize, ha='left', va='top',
             transform=fig.transFigure, linespacing=1.4, color='#333333')


def _add_tp_legend(ax):
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                      markersize=5, label=f'TP{tp}')
               for tp, c in TP_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc='upper right')


# ── Figure 1: Noise model ─────────────────────────────────────────────────────

def fig_noise_model(noise_df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8),
                              gridspec_kw={'width_ratios': [2, 1]})

    ax = axes[0]
    ax.errorbar(noise_df['tract_length'], noise_df['epsilon'],
                yerr=noise_df['epsilon_se'], fmt='o', ms=3, color='#555555',
                elinewidth=0.5, capsize=2, label='Observed ε (binned)')
    ax.plot(noise_df['tract_length'], noise_df['epsilon_smoothed'],
            color='#e41a1c', lw=1.5, label='Smoothed ε(L)')
    for y, lbl, ls in [(0.05, 'ε = 0.05', '--'), (0.10, 'ε = 0.10', ':'),
                        (0.25, 'ε = 0.25', '-.')]:
        ax.axhline(y, color='#888888', ls=ls, lw=0.7, label=lbl)
    ax.axvspan(noise_df['tract_length'].min(), 12.5,
               alpha=0.08, color='#2166ac', label='Clean-tract zone (ε ≤ 0.05)')
    ax.set_xlabel('polyA tract reference length (nt)')
    ax.set_ylabel('Per-observation error rate ε')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('A  Noise model ε(L)', loc='left', fontsize=7, fontweight='bold')

    ax2 = axes[1]
    ax2.bar(noise_df['tract_length'], noise_df['n_pairs'] / 1e6,
            color='#aec7e8', edgecolor='none', width=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Tract length (nt)')
    ax2.set_ylabel('Tag-site pairs (×10⁶)')
    ax2.set_title('B  Data weight', loc='left', fontsize=7, fontweight='bold')

    fig.tight_layout(pad=0.8)
    _add_caption(fig,
        legend=("(A) Per-observation error rate ε (black dots ± SE) estimated from the "
                "inter-tag-site pair mismatch rate across all insertions, with Gaussian-kernel "
                "smoothed curve (red). Horizontal dashed lines: ε = 0.05, 0.10, 0.25. "
                "Blue shading: clean-tract zone (ε ≤ 0.05, L ≤ 12 nt). "
                "(B) Number of tag-site pairs per length bin (log scale), showing data weight."),
        conclusion=("Measurement noise increases steeply with tract length, from <0.1 % at "
                    "L = 6 nt to ~30 % at L ≥ 25 nt. Tracts with L ≤ 12 nt (ε ≤ 0.05) "
                    "provide reliable single-observation calls and define the clean-tract class "
                    "used in downstream analysis."))
    _savefig(fig, os.path.join(out_dir, 'figures', 'fig1_noise_model.pdf'))


# ── Figure 3: Per-sample rates ────────────────────────────────────────────────

def fig_per_sample_rates(sample_summary, per_cell, out_dir):
    ts_ss = sample_summary[sample_summary['is_timeseries']].copy()
    if ts_ss.empty:
        ts_ss = sample_summary.copy()
        ts_ss['time_point'] = np.nan
        ts_ss = ts_ss.sort_values('sample')
    else:
        ts_ss = ts_ss.sort_values(['time_point', 'donor', 'sample'])
    sample_order = ts_ss['sample'].tolist()

    pc_ts = per_cell[per_cell['sample'].isin(sample_order)].copy()
    pc_ts = pc_ts.merge(ts_ss[['sample', 'time_point']], on='sample', how='left')
    pc_ts['sample'] = pd.Categorical(pc_ts['sample'], categories=sample_order, ordered=True)
    pc_ts['tp_color'] = pc_ts['time_point'].apply(
        lambda t: TP_COLORS.get(int(t), '#aaaaaa') if pd.notna(t) else '#aaaaaa')

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    ax = axes[0]
    data = [pc_ts[pc_ts['sample'] == s]['mutation_rate'].dropna().values
            for s in sample_order]
    data = [d if len(d) > 0 else np.array([0.0]) for d in data]
    parts = ax.violinplot(data, positions=range(len(sample_order)),
                          widths=0.7, showmedians=True)
    for i, (pc2, s) in enumerate(zip(parts['bodies'], sample_order)):
        tp_val = ts_ss[ts_ss['sample'] == s]['time_point'].iloc[0]
        tp_int = int(tp_val) if pd.notna(tp_val) else 0
        pc2.set_facecolor(TP_COLORS.get(tp_int, '#aaaaaa'))
        pc2.set_alpha(0.6)
    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels([s.replace(' (', '\n(') for s in sample_order],
                       rotation=90, fontsize=5)
    ax.set_ylabel('Mutation rate (corrected/tested)')
    ax.set_title('A  Per-cell distribution', loc='left', fontsize=7, fontweight='bold')
    _add_tp_legend(ax)

    ax = axes[1]
    for i, row in ts_ss.reset_index(drop=True).iterrows():
        tp = int(row['time_point']) if not pd.isna(row['time_point']) else 0
        ax.errorbar(i, row['mean_mutation_rate'], yerr=row['se_mutation_rate'],
                    fmt='o', ms=4, color=TP_COLORS.get(tp, '#aaaaaa'),
                    elinewidth=0.8, capsize=2)
    ax.set_xticks(range(len(ts_ss)))
    ax.set_xticklabels([s.replace(' (', '\n(') for s in sample_order],
                       rotation=90, fontsize=5)
    ax.set_ylabel('Mean mutation rate ± SE')
    ax.set_title('B  Per-sample means', loc='left', fontsize=7, fontweight='bold')
    _add_tp_legend(ax)

    ax = axes[2]
    sd_df = pc_ts.groupby('sample')['mutation_rate'].agg(['mean', 'std']).reset_index()
    sd_df = sd_df.merge(ts_ss[['sample', 'time_point']], on='sample')
    for _, row in sd_df.iterrows():
        tp = int(row['time_point']) if not pd.isna(row['time_point']) else 0
        ax.scatter(row['mean'], row['std'],
                   color=TP_COLORS.get(tp, '#aaaaaa'), s=18, lw=0)
    ax.set_xlabel('Mean mutation rate')
    ax.set_ylabel('SD of mutation rate')
    ax.set_title('C  Within-sample variability', loc='left', fontsize=7, fontweight='bold')
    _add_tp_legend(ax)

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'fig3_per_sample_rates.pdf'))


# ── Figure 4: Time-series ─────────────────────────────────────────────────────

def fig_timeseries(ts, tp_stats, ts_cells, out_dir):
    from scipy import stats as _st
    import statsmodels.api as sm

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    ax = axes[0]
    markers = {'Erythroid': 'o', 'Myeloid': 's', 'Mixed': '^'}
    for _, row in ts.iterrows():
        tp = float(row['time_point']) if not pd.isna(row['time_point']) else np.nan
        donor = row.get('donor', '')
        ct    = row.get('cell_type', 'Mixed')
        jitter = np.random.uniform(-0.08, 0.08)
        ax.errorbar(tp + jitter, row['mean_mutation_rate'],
                    yerr=row['se_mutation_rate'],
                    fmt=markers.get(ct, 'o'), ms=5,
                    color=DONOR_COLORS.get(donor, '#555555'),
                    elinewidth=0.7, capsize=2, alpha=0.85)

    x = ts['time_point'].values.astype(float)
    y = ts['mean_mutation_rate'].values
    slope, intercept, r, p2, _ = _st.linregress(x, y)
    p1s = p2 / 2 if slope > 0 else 1.0
    x_line = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    y_line = intercept + slope * x_line
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit()
    pred = fit.get_prediction(sm.add_constant(x_line))
    ci = pred.conf_int(alpha=0.05)
    ax.plot(x_line, y_line, color='#333333', lw=1, zorder=2)
    ax.fill_between(x_line, ci[:, 0], ci[:, 1], alpha=0.12, color='#333333')
    tp_ticks = sorted(ts['time_point'].unique())
    ax.set_xticks(tp_ticks)
    ax.set_xlabel('Time point')
    ax.set_ylabel('Mutation rate (corrected/tested)')
    ax.annotate(f'slope={slope:.4f}\none-sided p={p1s:.3f}',
                xy=(0.05, 0.95), xycoords='axes fraction', va='top', fontsize=6)
    ax.set_title('A  Per-sample regression', loc='left', fontsize=7, fontweight='bold')

    ax = axes[1]
    for _, row in tp_stats.iterrows():
        tp = int(row['time_point'])
        ax.bar(tp, row['tp_mean'], color=TP_COLORS.get(tp, '#aaa'),
               yerr=row['tp_se'], capsize=3, error_kw={'lw': 0.8},
               width=0.5, edgecolor='#333', linewidth=0.4)
    ax.set_xticks(tp_ticks)
    ax.set_ylabel('Mean mutation rate ± SE')
    ax.set_title('B  Time-point means', loc='left', fontsize=7, fontweight='bold')

    ax = axes[2]
    ts_cells = ts_cells.dropna(subset=['time_point', 'mutation_rate'])
    for tp, grp in ts_cells.groupby('time_point'):
        tp = int(tp)
        rates = grp['mutation_rate'].values
        from scipy.stats import gaussian_kde
        if len(rates) < 10:
            continue
        kde = gaussian_kde(rates, bw_method=0.3)
        x_kde = np.linspace(rates.min(), rates.max(), 200)
        ax.plot(x_kde, kde(x_kde), color=TP_COLORS.get(tp, '#aaa'),
                lw=1.2, label=f'TP{tp}')
    ax.set_xlabel('Per-cell mutation rate')
    ax.set_ylabel('Density')
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('C  Cell-level distribution', loc='left', fontsize=7, fontweight='bold')

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'fig4_timeseries.pdf'))


# ── ED Figure 1: Allele tiers ─────────────────────────────────────────────────

def fig_allele_tiers(mutations, allele_calls, out_dir):
    m = mutations.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))

    ax = axes[0]
    tier_counts = m.groupby('confidence_tier')['ins_key'].nunique()
    tier_labels = {1: 'SNP-phased', 2: 'Clean-seq\n(δ<1%)',
                   3: 'Noisy-seq\n(δ≥1%)', 4: 'Biallelic-peak',
                   5: 'Homozygous'}
    tiers = sorted(tier_counts.index)
    ax.bar([tier_labels.get(t, str(t)) for t in tiers],
           [tier_counts[t] for t in tiers],
           color=[TIER_COLORS.get(t, '#aaa') for t in tiers],
           edgecolor='#333', linewidth=0.4)
    ax.set_ylabel('Insertions (unique ins_key)')
    ax.set_xlabel('Confidence tier')
    ax.set_title('A  Insertion counts per tier', loc='left', fontsize=7, fontweight='bold')

    ax = axes[1]
    cond_probs = []
    for ins_info in allele_calls.values():
        for allele_info in ins_info.values():
            for p in allele_info.get('conditional_probs', {}).values():
                cond_probs.append(float(p))
    cond_probs = np.array(cond_probs)
    ax.hist(cond_probs, bins=50, color='#4393c3', edgecolor='none', alpha=0.8)
    ax.axvline(0.01, color='#e41a1c', ls='--', lw=0.8, label='1% threshold (Tier 2/3 split)')
    ax.set_xscale('symlog', linthresh=0.001)
    ax.set_xlabel('Allele bleed-through rate (conditional_prob)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('B  Bleed-through distribution', loc='left', fontsize=7, fontweight='bold')

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed1_allele_tiers.pdf'))


# ── ED Figure 2: Cross-tract consistency ─────────────────────────────────────

def fig_cross_tract(annotated, out_dir):
    m = annotated.reset_index()
    if 'n_tested_tracts' not in m.columns or 'frac_nonref' not in m.columns:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.axis('off')
        ax.text(0.5, 0.5,
                'Cross-tract columns not available.\nRun add_cross_tract_consistency() first.',
                ha='center', va='center', fontsize=8, transform=ax.transAxes)
        _savefig(fig, os.path.join(out_dir, 'figures', 'ed2_cross_tract.pdf'))
        return
    multi = m[m['n_tested_tracts'] >= 2].copy()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    ax = axes[0]
    pattern_col = np.where(multi['misassign_pattern'], 'Mis-assign',
                   np.where(multi['is_isolated'], 'Isolated', 'Other'))
    multi['pattern'] = pattern_col
    colors_pat = {'Mis-assign': '#e41a1c', 'Isolated': '#4daf4a', 'Other': '#aaaaaa'}
    for pat, grp in multi.groupby('pattern'):
        ax.hist(grp['frac_nonref'].dropna(), bins=30, alpha=0.65,
                color=colors_pat[pat], label=pat, edgecolor='none')
    ax.set_xlabel('Fraction of tested tracts non-reference')
    ax.set_ylabel('Count')
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('A  Cross-tract pattern distribution', loc='left', fontsize=7,
                 fontweight='bold')

    ax = axes[1]
    rates = (multi.groupby('confidence_tier')
             [['misassign_pattern', 'is_isolated']]
             .mean()
             .reset_index())
    x = np.arange(len(rates))
    w = 0.35
    ax.bar(x - w/2, rates['misassign_pattern'], width=w, label='Mis-assign',
           color='#e41a1c', edgecolor='#333', linewidth=0.4)
    ax.bar(x + w/2, rates['is_isolated'], width=w, label='Isolated',
           color='#4daf4a', edgecolor='#333', linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Tier {int(t)}' for t in rates['confidence_tier']])
    ax.set_ylabel('Fraction of candidates')
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('B  Pattern rate by tier', loc='left', fontsize=7, fontweight='bold')

    ax = axes[2]
    mis = multi[multi['misassign_pattern']].copy()
    if len(mis) > 0:
        ax.hist(mis['delta_length'],
                bins=range(int(mis['delta_length'].min()) - 1,
                           int(mis['delta_length'].max()) + 2),
                color='#e41a1c', edgecolor='#333', linewidth=0.4)
    ax.axvline(0, color='#333', lw=0.8)
    ax.set_xlabel('delta_length')
    ax.set_ylabel('Mis-assign candidates')
    ax.set_title('C  Mis-assign Δ distribution', loc='left', fontsize=7, fontweight='bold')

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed2_cross_tract.pdf'))


# ── ED Figure 3: Coverage distribution ───────────────────────────────────────

def fig_coverage(per_cell, sample_summary, min_tested, out_dir):
    import re
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    ax = axes[0]
    vals = per_cell['n_tested_tracts'].dropna()
    ax.hist(vals, bins=60, color='#4393c3', edgecolor='none', alpha=0.8)
    ax.axvline(min_tested, color='#e41a1c', ls='--', lw=0.8,
               label=f'Filter: ≥{min_tested}')
    ax.set_xlabel('n_tested_tracts per cell')
    ax.set_ylabel('Cell count')
    ax.legend(fontsize=6, frameon=False)
    frac = (vals >= min_tested).mean()
    ax.set_title(f'A  Coverage ({frac:.0%} pass filter)', loc='left', fontsize=7,
                 fontweight='bold')

    ax = axes[1]
    ts_ss = sample_summary[sample_summary['is_timeseries']].copy()
    if ts_ss.empty:
        ts_ss = sample_summary.copy()
        ts_ss['time_point'] = np.nan
        ts_ss = ts_ss.sort_values('sample')
    else:
        ts_ss = ts_ss.sort_values(['time_point', 'donor'])
    ts_cells_pc = per_cell[per_cell['sample'].isin(ts_ss['sample'])].copy()
    ts_cells_pc = ts_cells_pc.merge(
        ts_ss[['sample', 'time_point']], on='sample', how='left')
    sample_order = ts_ss['sample'].tolist()
    data_by_sample = [
        ts_cells_pc[ts_cells_pc['sample'] == s]['n_tested_tracts'].dropna().values
        for s in sample_order]
    data_by_sample = [d if len(d) > 0 else np.array([0.0]) for d in data_by_sample]
    parts = ax.violinplot(data_by_sample, positions=range(len(sample_order)),
                          widths=0.7, showmedians=True)
    for i, (pc2, s) in enumerate(zip(parts['bodies'], sample_order)):
        tp = ts_ss[ts_ss['sample'] == s]['time_point'].iloc[0]
        pc2.set_facecolor(TP_COLORS.get(int(tp) if not pd.isna(tp) else 0, '#aaa'))
        pc2.set_alpha(0.6)

    def _abbrev(name):
        s = re.sub(r'\s*\(\d+/\d+ sort\)\s*', '', name).strip()
        s = re.sub(r'\bSet (\d+)\b', r'S\1', s)
        return s.replace('Erythroid', 'E').replace('Myeloid', 'M')

    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels([_abbrev(s) for s in sample_order], rotation=90, fontsize=5)
    ax.axhline(min_tested, color='#e41a1c', ls='--', lw=0.7)
    ax.set_ylabel('n_tested_tracts')
    ax.set_title('B  Per-sample coverage', loc='left', fontsize=7, fontweight='bold')

    ax = axes[2]
    pc_ts = ts_cells_pc[ts_cells_pc['n_tested_tracts'] >= min_tested].copy()
    for _, row in ts_ss.iterrows():
        tp = int(row['time_point']) if not pd.isna(row['time_point']) else 0
        sub = pc_ts[pc_ts['sample'] == row['sample']]
        if sub.empty:
            continue
        ax.scatter(sub['n_tested_tracts'], sub['mutation_rate'],
                   s=1, alpha=0.3, color=TP_COLORS.get(tp, '#aaa'), rasterized=True)
    ax.set_xlabel('n_tested_tracts per cell')
    ax.set_ylabel('Mutation rate')
    ax.set_title('C  Coverage vs rate', loc='left', fontsize=7, fontweight='bold')
    _add_tp_legend(ax)

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed3_coverage.pdf'))


# ── ED Figure 5: p-value exploration ─────────────────────────────────────────

def fig_pvalue_exploration(regression_rows, out_dir):
    if not regression_rows:
        return
    df = pd.DataFrame(regression_rows).sort_values('p_two_sided')

    fig, ax = plt.subplots(figsize=(5, max(3, len(df) * 0.28)))
    for i, (_, row) in enumerate(df.iterrows()):
        p = row.get('p_one_sided' if row.get('is_one_sided') else 'p_two_sided', np.nan)
        color = '#2166ac' if row.get('is_one_sided', False) else '#d6604d'
        if pd.notna(p):
            ax.scatter(p, i, color=color, s=20, zorder=3)
            ax.plot([p, p], [i - 0.3, i + 0.3], color=color, lw=1.5)

    ax.axvline(0.05, color='#e41a1c', ls='--', lw=0.8, label='p = 0.05')
    ax.set_xscale('log')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['label'].tolist(), fontsize=6)
    ax.set_xlabel('p-value')
    ax.set_title('ED5  p-value exploration', loc='left', fontsize=7, fontweight='bold')
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#2166ac',
                      markersize=5, label='One-sided'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='#d6604d',
                      markersize=5, label='Two-sided')]
    ax.legend(handles=handles, fontsize=6, frameon=False)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed5_pvalue_exploration.pdf'))


# ── ED Figure 6: SNP concordance ─────────────────────────────────────────────

def fig_snp_concordance(phase_allele_snv, phase_biallele_snv, allele_calls, out_dir):
    seq2_keys = {k for k, v in allele_calls.items() if len(v) == 2}

    def _build_df(phase_dict, subset_keys, method_label):
        rows = []
        for ins_key, sites in phase_dict.items():
            if subset_keys is not None and ins_key not in subset_keys:
                continue
            best_chi2 = -1
            best = {}
            for entry in sites.values():
                allele_to_snp, chi2, p, dof, _ = entry
                if chi2 is None or p is None:
                    continue
                alleles = list(allele_to_snp.keys())
                if len(alleles) != 2:
                    continue
                c0, c1 = allele_to_snp[alleles[0]], allele_to_snp[alleles[1]]
                bases = sorted(set(list(c0.keys()) + list(c1.keys())))
                if len(bases) < 2:
                    continue
                b0, b1 = bases[0], bases[1]
                a00 = c0.get(b0, 0); a01 = c0.get(b1, 0)
                a10 = c1.get(b0, 0); a11 = c1.get(b1, 0)
                total = a00 + a01 + a10 + a11
                if total == 0 or chi2 <= best_chi2:
                    continue
                best_chi2 = chi2
                best = {'ins_key': ins_key, 'p': p, 'chi2': chi2,
                        'concordance': max(a00 + a11, a01 + a10) / total,
                        'n_reads': total, 'method': method_label}
            if best:
                rows.append(best)
        return pd.DataFrame(rows)

    seq_df = _build_df(phase_allele_snv, seq2_keys, 'Sequence-based\n(Tier 2/3)')
    pbc_df = _build_df(phase_biallele_snv, None,     'Peak-based\n(Tier 4)')

    COLORS = {'Sequence-based\n(Tier 2/3)': '#2166ac',
              'Peak-based\n(Tier 4)':        '#d6604d'}
    LOW_CONC = 0.80
    SIG      = 0.05

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    for df in [seq_df, pbc_df]:
        if df.empty:
            continue
        label = df['method'].iloc[0]
        n_sig = (df['p'] < SIG).sum()
        ax.hist(df['concordance'], bins=bins, alpha=0.6, color=COLORS[label],
                label=f'{label}  (n={len(df)}, sig={n_sig})', edgecolor='none')
    ax.axvline(LOW_CONC, color='#333', lw=0.8, ls='--', label=f'{int(LOW_CONC*100)}% threshold')
    ax.set_xlabel('Concordance (best het site)')
    ax.set_ylabel('Insertions')
    ax.set_title('A  Concordance distribution', loc='left', fontsize=7, fontweight='bold')
    ax.legend(fontsize=5.5, frameon=False)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    ax = axes[1]
    for df in [seq_df, pbc_df]:
        if df.empty:
            continue
        label = df['method'].iloc[0]
        sig = df[df['p'] < SIG]
        ns  = df[df['p'] >= SIG]
        ax.scatter(sig['concordance'], -np.log10(sig['p'].clip(lower=1e-320)),
                   color=COLORS[label], s=10, alpha=0.7, lw=0, label=label + ' p<0.05')
        if len(ns):
            ax.scatter(ns['concordance'], -np.log10(ns['p'].clip(lower=1e-320)),
                       color=COLORS[label], s=10, alpha=0.35, lw=0, marker='x',
                       label=label + ' n.s.')
    ax.axvline(LOW_CONC, color='#333', lw=0.8, ls='--')
    ax.axhline(-np.log10(SIG), color='#999', lw=0.6, ls=':')
    ax.set_xlabel('Concordance')
    ax.set_ylabel('−log₁₀(p)')
    ax.set_title('B  Concordance vs significance', loc='left', fontsize=7, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=5, frameon=False)

    ax = axes[2]
    for df in [seq_df, pbc_df]:
        if df.empty:
            continue
        label = df['method'].iloc[0]
        sig = df[df['p'] < SIG]['concordance'].sort_values()
        if len(sig) == 0:
            continue
        cdf = np.arange(1, len(sig) + 1) / len(sig)
        ax.plot(sig, cdf, color=COLORS[label], lw=1.2, label=f'{label}  n={len(sig)}')
    ax.axvline(LOW_CONC, color='#333', lw=0.8, ls='--', label='80%')
    ax.axvline(0.90,     color='#aaa', lw=0.8, ls='--', label='90%')
    ax.set_xlabel('Concordance')
    ax.set_ylabel('Cumulative fraction')
    ax.set_title('C  CDF (significant only)', loc='left', fontsize=7, fontweight='bold')
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=5.5, frameon=False)

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed6_snp_concordance.pdf'))


# ── ED Figure 7: Length-adjusted vs standard mutation rate ───────────────────

def fig_lenadj_rate(ts, sample_summary, out_dir):
    if 'mean_mutation_rate_lenadj' not in ts.columns:
        return
    ts = ts.dropna(subset=['mean_mutation_rate_lenadj', 'mean_mutation_rate']).copy()
    if ts.empty:
        return

    _TP = {1: '#4393c3', 2: '#d6604d', 3: '#1a9850'}
    tp_vals = sorted(ts['time_point'].dropna().unique())

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

    ax = axes[0]
    for tp in tp_vals:
        sub = ts[ts['time_point'] == tp]
        ax.scatter(sub['mean_mutation_rate'], sub['mean_mutation_rate_lenadj'],
                   color=_TP.get(int(tp), '#aaa'), s=18, lw=0, label=f'TP{int(tp)}')
    lims = [min(ts['mean_mutation_rate'].min(), ts['mean_mutation_rate_lenadj'].min()) * 0.97,
            max(ts['mean_mutation_rate'].max(), ts['mean_mutation_rate_lenadj'].max()) * 1.03]
    ax.plot(lims, lims, color='#999', lw=0.6, ls='--')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Standard rate (1/n_tested_tracts)')
    ax.set_ylabel('Length-adjusted rate (1/Σε/med_ε)')
    ax.set_title('A  Rate comparison per sample', loc='left', fontsize=7, fontweight='bold')
    ax.legend(fontsize=6, frameon=False)

    ax = axes[1]
    for metric, label, ls in [
        ('mean_mutation_rate',        'Standard',        '-'),
        ('mean_mutation_rate_lenadj', 'Length-adjusted', '--'),
    ]:
        tp1_mean = ts[ts['time_point'] == min(tp_vals)][metric].mean()
        if tp1_mean == 0:
            continue
        for donor, grp in ts.groupby('donor'):
            grp = grp.sort_values('time_point')
            ax.plot(grp['time_point'], grp[metric] / tp1_mean,
                    ls=ls, lw=0.9, alpha=0.6, color='#555')
        tp_means = ts.groupby('time_point')[metric].mean() / tp1_mean
        ax.plot(tp_means.index, tp_means.values, ls=ls, lw=1.8, color='#222', label=label)
    ax.set_xlabel('Time point')
    ax.set_ylabel('Rate / TP1 mean')
    ax.set_title('B  Trend (normalised to TP1)', loc='left', fontsize=7, fontweight='bold')
    ax.legend(fontsize=6, frameon=False)
    ax.set_xticks(sorted(ts['time_point'].unique()))

    ax = axes[2]
    ts['_ratio'] = ts['mean_mutation_rate_lenadj'] / ts['mean_mutation_rate']
    for tp in tp_vals:
        sub = ts[ts['time_point'] == tp]
        ax.scatter([tp] * len(sub), sub['_ratio'],
                   color=_TP.get(int(tp), '#aaa'), s=18, lw=0, label=f'TP{int(tp)}')
    tp_ratio = ts.groupby('time_point')['_ratio'].mean()
    ax.plot(tp_ratio.index, tp_ratio.values, color='#222', lw=1.2, zorder=5)
    ax.axhline(1.0, color='#aaa', lw=0.6, ls='--')
    ax.set_xlabel('Time point')
    ax.set_ylabel('Length-adjusted / standard rate')
    ax.set_title('C  Ratio per sample', loc='left', fontsize=7, fontweight='bold')
    ax.set_xticks(sorted(ts['time_point'].unique()))
    ax.legend(fontsize=6, frameon=False)

    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed7_lenadj_rate.pdf'))


# ── Clone analysis functions ──────────────────────────────────────────────────

def _clone_counts_one_ins(ins_key, df, dna_to_clone, value_key, min_tag_sites):
    """Worker for clone_cell_counts_from_histograms (one insertion)."""
    df_r = df.reset_index()
    if value_key in df_r.columns:
        df_r = df_r[df_r[value_key].notna()]
    df_r['clone_group'] = df_r['cell'].map(dna_to_clone)
    df_r = df_r[df_r['clone_group'].notna()]
    if df_r.empty:
        return {}
    obs = (df_r.groupby(['cell', 'a_key', 't_key', 'clone_group'])['tag_site']
           .count()
           .reset_index(name='n_obs'))
    obs = obs[obs['n_obs'] >= min_tag_sites]
    counts = {}
    for _, row in obs.iterrows():
        key = (ins_key, int(row['a_key']), int(row['t_key']), int(row['clone_group']))
        counts[key] = counts.get(key, 0) + 1
    return counts


def clone_cell_counts_from_histograms(histogram_dfs, dna_to_clone,
                                      min_tag_sites=2, value_key='hist_peak',
                                      n_jobs=1):
    """
    Count, for each (ins_key, a_key, t_key, clone_group), how many cells have
    >= min_tag_sites valid observations.
    Returns dict {(ins_key, a_key, t_key, clone_group): n_cells}.
    """
    from joblib import Parallel, delayed as _delayed

    results = Parallel(n_jobs=n_jobs)(
        _delayed(_clone_counts_one_ins)(ins_key, df, dna_to_clone, value_key, min_tag_sites)
        for ins_key, df in histogram_dfs.items()
    )
    counts = {}
    for partial in results:
        for k, v in partial.items():
            counts[k] = counts.get(k, 0) + v
    return counts


def annotate_clone_consistency(mutations_df, clone_cell_counts, dna_to_clone):
    """Add clone-level consistency columns to the mutations DataFrame."""
    m = mutations_df.reset_index().copy()
    m['clone_group'] = m['cell'].map(dna_to_clone).astype('Int64')

    with_clone = m[m['clone_group'].notna()].copy()
    grp_cols = ['ins_key', 'a_key', 't_key', 'clone_group', 'cell_length']
    agg = (with_clone.groupby(grp_cols)
           .agg(n_clone_calling=('cell', 'count'),
                n_clone_is_mutated=('is_mutated', 'sum'))
           .reset_index())
    m = m.merge(agg, on=grp_cols, how='left')

    def _lookup_clone_total(row):
        if pd.isna(row['clone_group']):
            return np.nan
        return clone_cell_counts.get(
            (row['ins_key'], int(row['a_key']), int(row['t_key']), int(row['clone_group'])), 0)

    m['n_clone_cells']  = m.apply(_lookup_clone_total, axis=1)
    m['clone_fraction'] = m['n_clone_calling'] / m['n_clone_cells'].replace(0, np.nan)
    m['all_agree']      = m['n_clone_calling'] == m['n_clone_cells']
    return m.set_index(['ins_key', 'cell', 'a_key', 't_key'])


def summarise_clone_consistency(annotated, min_clone_size=2):
    """Summary statistics per delta_bucket for calibration."""
    df = annotated.reset_index()
    df['delta_bucket'] = df['delta_length'].abs().apply(
        lambda d: '±1' if d == 1 else ('±2' if d == 2 else '|Δ|≥3'))

    rows = []
    for bucket, grp in df.groupby('delta_bucket'):
        with_clone = grp[grp['clone_group'].notna()]
        sizege2 = with_clone[with_clone['n_clone_cells'] >= min_clone_size]
        rows.append(dict(
            delta_bucket=bucket,
            n_candidates=len(grp),
            n_is_mutated=grp['is_mutated'].sum(),
            n_with_clone=len(with_clone),
            n_clone_size_ge2=len(sizege2),
            n_all_agree=sizege2['all_agree'].sum(),
            frac_all_agree=(sizege2['all_agree'].mean() if len(sizege2) > 0 else np.nan),
        ))
    summary = pd.DataFrame(rows).set_index('delta_bucket')

    grp_cols = ['ins_key', 'a_key', 't_key', 'clone_group', 'cell_length']
    clone_mut = (
        df[df['clone_group'].notna()]
        .groupby(grp_cols)
        .agg(
            ref_length=('ref_length', 'first'),
            delta_length=('delta_length', 'first'),
            delta_bucket=('delta_bucket', 'first'),
            n_clone_cells=('n_clone_cells', 'first'),
            n_clone_calling=('n_clone_calling', 'first'),
            n_clone_is_mutated=('n_clone_is_mutated', 'first'),
            clone_fraction=('clone_fraction', 'first'),
            all_agree=('all_agree', 'first'),
            min_p_fp=('p_false_positive', 'min'),
            epsilon=('epsilon', 'first'),
            background_rate=('background_rate', 'first'),
            confidence_tier=('confidence_tier', 'first'),
            snp_concordance=('snp_concordance', 'first'),
            snp_concordance_p=('snp_concordance_p', 'first'),
            snp_concordance_low=('snp_concordance_low', 'first'),
        )
        .reset_index()
    )
    return summary, clone_mut


def add_clone_pvalue(clone_mut, annotated, tested_cells):
    """Fisher's exact test for clone enrichment of L_mut with BH-FDR correction."""
    from scipy.stats import fisher_exact as _fisher_exact
    from statsmodels.stats.multitest import multipletests as _mct

    df = clone_mut.copy()
    df['cell_length'] = df['cell_length'].astype(int)

    ann_r = annotated.reset_index()
    ann_r['cell_length'] = ann_r['cell_length'].astype(int)
    K_df = (ann_r.groupby(['ins_key', 'a_key', 't_key', 'cell_length'])['cell']
            .nunique().rename('K_calling').reset_index())
    df = df.merge(K_df, on=['ins_key', 'a_key', 't_key', 'cell_length'], how='left')

    tc = tested_cells.reset_index()[['ins_key', 'a_key', 't_key', 'n_cells_tested']]
    tc = tc.groupby(['ins_key', 'a_key', 't_key'], as_index=False)['n_cells_tested'].max()
    df = df.merge(tc, on=['ins_key', 'a_key', 't_key'], how='left')
    df = df.rename(columns={'n_cells_tested': 'N_tested'})

    df['K_outside'] = (df['K_calling'] - df['n_clone_calling']).clip(lower=0)
    df['N_outside'] = (df['N_tested']  - df['n_clone_cells']).clip(lower=0)

    def _run_fisher(k, n_clone, K_out, N_out):
        n_clone_other = int(n_clone) - int(k)
        N_out_other   = int(N_out)   - int(K_out)
        if n_clone_other < 0 or N_out_other < 0:
            return np.nan, np.nan
        try:
            odds, p = _fisher_exact([[int(k), n_clone_other],
                                      [int(K_out), N_out_other]], alternative='greater')
            return float(odds), float(p)
        except Exception:
            return np.nan, np.nan

    results = [_run_fisher(k, n, Ko, No)
               for k, n, Ko, No in zip(df['n_clone_calling'], df['n_clone_cells'],
                                        df['K_outside'], df['N_outside'])]
    df['odds_ratio'] = [r[0] for r in results]
    df['p_fisher']   = [r[1] for r in results]

    valid_mask = df['p_fisher'].notna()
    q_vals = np.full(len(df), np.nan)
    if valid_mask.any():
        _, q_adj, _, _ = _mct(df.loc[valid_mask, 'p_fisher'].values, alpha=0.05, method='fdr_bh')
        q_vals[valid_mask.values] = q_adj
    df['q_fisher'] = q_vals

    df['K_rate_outside'] = df['K_outside'] / df['N_outside'].replace(0, np.nan)
    max_nonclone_rate = df['epsilon'].clip(upper=0.10)
    df['p_clone_noise'] = df['background_rate'] ** df['n_clone_calling']
    df['is_clone_mutated'] = (
        (df['clone_fraction'] >= 0.9) &
        (df['n_clone_calling'] >= 2) &
        (df['q_fisher'] < 0.05) &
        (df['K_rate_outside'] <= max_nonclone_rate)
    )
    return df


def clone_size_breakdown(annotated, min_clone_cells=2):
    """Per-delta-bucket all_agree rate by clone size for geometric FPR fitting."""
    df = annotated.reset_index()
    df['delta_bucket'] = df['delta_length'].abs().apply(
        lambda d: '±1' if d == 1 else ('±2' if d == 2 else '|Δ|≥3'))
    breakdown = {}
    for bucket, grp in df.groupby('delta_bucket'):
        has_clone = grp[grp['clone_group'].notna() & (grp['n_clone_cells'] >= min_clone_cells)]
        by_size = (has_clone.groupby('n_clone_cells')['all_agree']
                   .agg(n_all_agree='sum', n_total='count')
                   .assign(rate=lambda x: x['n_all_agree'] / x['n_total']))
        breakdown[bucket] = by_size
    return breakdown


def fit_calibrated_fpr(breakdown):
    """Fit per-cell noise FPR via log-linear regression on clone-size data."""
    from scipy import stats as _st
    calibrated = {}
    for bucket, by_size in breakdown.items():
        valid = by_size[by_size['n_all_agree'] > 0].copy()
        if len(valid) >= 2:
            slope, *_ = _st.linregress(valid.index.values.astype(float),
                                        np.log(valid['rate'].values))
            p = float(np.clip(np.exp(slope), 0.0, 1.0))
        elif len(valid) == 1:
            k = float(valid.index[0])
            p = float(np.clip(valid['rate'].iloc[0] ** (1.0 / k), 0.0, 1.0))
        else:
            if len(by_size) > 0 and by_size['n_all_agree'].sum() > 0:
                agg_rate = by_size['n_all_agree'].sum() / by_size['n_total'].sum()
                p = float(np.clip(agg_rate ** (1.0 / float(by_size.index.min())), 0.0, 1.0))
            else:
                p = 0.05
        calibrated[bucket] = p
    return calibrated


# ── Clone figure functions ────────────────────────────────────────────────────

def fig_clone_calibration(breakdown, calibrated_fpr, out_dir):
    buckets = ['±1', '±2', '|Δ|≥3']
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharey=False)
    for ax, bucket in zip(axes, buckets):
        by_size = breakdown.get(bucket, pd.DataFrame())
        color = BUCKET_COLORS[bucket]
        if by_size.empty:
            ax.set_visible(False)
            continue
        valid = by_size[by_size['n_all_agree'] > 0]
        ax.scatter(valid.index, valid['rate'], s=10, color=color, zorder=3, label='Observed')
        p_noise = calibrated_fpr.get(bucket, np.nan)
        if not np.isnan(p_noise):
            k_range = np.linspace(by_size.index.min(), by_size.index.max(), 200)
            ax.plot(k_range, p_noise ** k_range, color=color, lw=1.2,
                    label=f'p_noise^k\n(p={p_noise:.3f})')
        ax.set_yscale('log')
        ax.set_xlabel('Clone size k')
        if bucket == '±1':
            ax.set_ylabel('Fraction all-agree')
        ax.set_title(f'{bucket}', fontsize=7, fontweight='bold')
        ax.legend(fontsize=6, frameon=False)
        ax.axhline(0.05, color='grey', ls='--', lw=0.6)
    fig.suptitle('Figure 2  Clone-consistency calibration', fontsize=7,
                 fontweight='bold', x=0.02, ha='left')
    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'fig2_clone_calibration.pdf'))


def fig_clone_mutations(clone_mut, out_dir):
    if clone_mut.empty or not clone_mut.get('is_clone_mutated', pd.Series(dtype=bool)).any():
        return
    cm = clone_mut.copy()
    p_col = 'p_fisher' if 'p_fisher' in cm.columns else 'p_clone_noise'
    cm['log10_p'] = -np.log10(cm[p_col].clip(lower=1e-10))
    cm['abs_delta'] = cm['delta_length'].abs()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5),
                              gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axes[0]
    tier_handles = []
    for tier, grp in cm.groupby('confidence_tier'):
        highlighted = grp[grp['is_clone_mutated']]
        dimmed = grp[~grp['is_clone_mutated']]
        color = TIER_COLORS.get(int(tier), '#aaa')
        if not dimmed.empty:
            ax.scatter(dimmed['abs_delta'], dimmed['log10_p'],
                       s=dimmed['n_clone_calling'] * 8,
                       color=color, alpha=0.20, edgecolors='none')
        if not highlighted.empty:
            ax.scatter(highlighted['abs_delta'], highlighted['log10_p'],
                       s=highlighted['n_clone_calling'] * 18,
                       color=color, alpha=0.9, edgecolors='#333', linewidths=0.5)
            tier_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markersize=5, label=f'Tier {int(tier)}'))
            for _, row in highlighted.iterrows():
                ax.annotate(f"clone {int(row['clone_group'])}",
                            xy=(row['abs_delta'], row['log10_p']),
                            xytext=(4, 2), textcoords='offset points',
                            fontsize=4.5, color='#333333', va='bottom')
    ax.axhline(-np.log10(0.05), color='#e41a1c', ls='--', lw=0.8, label='p = 0.05')
    ax.set_xlabel('|Δ length| (nt)')
    ax.set_ylabel(f'−log₁₀({p_col})')
    size_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#777',
                            markersize=np.sqrt(n * 18), alpha=0.7, label=f'n = {n}')
                    for n in [2, 3, 4]]
    ax.legend(handles=tier_handles + size_handles +
              [Line2D([0], [0], color='#e41a1c', ls='--', lw=0.8, label='p = 0.05')],
              fontsize=5.5, frameon=False, ncol=2,
              title='Tier / Clone size', title_fontsize=5.5)
    ax.set_title('A  Clone-consistent mutation calls', loc='left', fontsize=7, fontweight='bold')

    ax = axes[1]
    ax.axis('off')
    top = (clone_mut[clone_mut['is_clone_mutated']]
           .sort_values('p_fisher' if 'p_fisher' in clone_mut.columns else 'p_clone_noise')
           .head(15)
           [['clone_group', 'ins_key', 'delta_length', 'n_clone_cells',
             'n_clone_calling', 'p_clone_noise', 'confidence_tier']]
           .copy())
    top['p_clone_noise'] = top['p_clone_noise'].map('{:.2e}'.format)
    top['clone_group'] = top['clone_group'].astype(int)
    top.columns = ['Clone\nID', 'Insertion', 'Δ', 'Clone\nsize', 'Calling', 'p_clone', 'Tier']
    tbl = ax.table(cellText=top.values, colLabels=top.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(5.0)
    tbl.scale(1, 1.25)
    ax.set_title('B  Top clone-consistent events', loc='left', fontsize=7, fontweight='bold')
    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'fig5_clone_mutations.pdf'))


def fig_fpr_calibration(breakdown, calibrated_fpr, summary, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    buckets = list(calibrated_fpr.keys())
    ax.bar(buckets, [calibrated_fpr[b] for b in buckets],
           color=[BUCKET_COLORS.get(b, '#aaa') for b in buckets],
           edgecolor='#333', linewidth=0.4)
    ax.axhline(0.05, color='#333', ls='--', lw=0.7, label='5% reference')
    ax.set_ylabel('Calibrated per-cell FPR (p_noise)')
    ax.set_xlabel('Δ length bucket')
    ax.legend(fontsize=6, frameon=False)
    ax.set_title('A  Calibrated FPR by bucket', loc='left', fontsize=7, fontweight='bold')

    ax = axes[1]
    ax.axis('off')
    disp = summary[['n_candidates', 'n_is_mutated', 'n_clone_size_ge2',
                     'n_all_agree', 'frac_all_agree']].copy()
    disp['frac_all_agree'] = disp['frac_all_agree'].map('{:.3f}'.format)
    disp.columns = ['Cands', 'is_mut', 'Clone≥2', 'All-agree', 'Frac-agree']
    tbl = ax.table(cellText=disp.values, rowLabels=disp.index,
                   colLabels=disp.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.3)
    ax.set_title('B  Clone consistency summary', loc='left', fontsize=7, fontweight='bold')
    fig.tight_layout(pad=0.8)
    _savefig(fig, os.path.join(out_dir, 'figures', 'ed4_fpr_calibration.pdf'))

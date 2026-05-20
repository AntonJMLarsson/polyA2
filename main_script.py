from support_functions import *
import argparse
import json
import os
from joblib import delayed,Parallel
import pandas as pd
import pickle


def _is_cached(path, overwrite):
    """Return True iff the output exists and we should not recompute it."""
    return os.path.exists(path) and not overwrite


def _warn_skip(path):
    print('WARNING: {} exists, using cached version. Use --overwrite to recompute.'.format(path))


def _load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def _save_pickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description='Find allelic variants and polyA tract lengths', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i','--input',metavar='input', type=str, help='Input .bam file')
    parser.add_argument('--insertions', metavar='output', type=str, help='Input .csv file with insertion information')
    parser.add_argument('-t', '--threads', metavar='threads', type=int, default=1, help='Number of threads')
    parser.add_argument('-s', '--samplesheet', metavar='samplesheet', type=str, help='Samplesheet, needs BCSet column')
    parser.add_argument('-p', '--prefix', metavar='prefix', type=str, help='Output prefix')
    parser.add_argument('--threshold', type=int, default=10, help='Read threshold to include.')
    parser.add_argument('--min_len', type=int, default=6, help='Minimum length to classify a polyA as a tract.')
    parser.add_argument('--break_len', type=int, default=0, help='Number of non-A bases allowed to interrupt a polyA tract.')
    parser.add_argument('--overwrite', action='store_true', help='Recompute and overwrite output files even if they already exist.')
    args = parser.parse_args()

    bamfile = args.input
    threads = args.threads
    samplesheet_file = args.samplesheet
    prefix = args.prefix
    min_len = args.min_len
    break_len = args.break_len
    overwrite = args.overwrite

    insertion_information = pd.read_csv(args.insertions, index_col = 0).T.to_dict()
    samplesheet = pd.read_csv(samplesheet_file, index_col=0)
    samplesheet_DNA = samplesheet[(samplesheet['Modality'] == 'DNA')]
    cell_BC_set = set(samplesheet_DNA.index)

    res_dict_path = '{}_res_dict.pkl'.format(prefix)
    allele_calls_path = '{}_allele_calls.pkl'.format(prefix)
    histogram_path = '{}_histogram_dfs.pkl'.format(prefix)
    peak_based_path = '{}_peak_based_calling.pkl'.format(prefix)
    snp_cpm_path = '{}_snp_cpm.pkl'.format(prefix)
    phase_allele_path = '{}_phase_allele_snv.pkl'.format(prefix)
    phase_biallele_path = '{}_phase_biallele_snv.pkl'.format(prefix)
    summary_path = '{}_insertion_summary.csv'.format(prefix)

    ## Stage 1: per-insertion allele detection.
    if _is_cached(res_dict_path, overwrite):
        _warn_skip(res_dict_path)
        res_dict = _load_pickle(res_dict_path)
    else:
        res = Parallel(n_jobs=threads, verbose = 3)(delayed(get_alleles_by_tagmentation_site)(bamfile, cell_BC_set, insertion_dict,key, min_len, break_len) for key, insertion_dict in insertion_information.items())
        res_dict = {t[0]: t[1] for t in res}
        _save_pickle(res_dict_path, res_dict)

    ## Stage 2: make a final decision on which sequence summaries represent allele variants.
    if _is_cached(allele_calls_path, overwrite):
        _warn_skip(allele_calls_path)
        allele_calls = _load_pickle(allele_calls_path)
    else:
        allele_calls = {}
        for ins_key, info_dict in res_dict.items():
            allele_counter = info_dict['allele_counter']
            tag_site_sum = sum(allele_counter.values())
            allele_key_dict = {}
            for allele_key, count in allele_counter.items():
                fraction_tags = count/tag_site_sum
                if fraction_tags < 0.1:
                    continue
                allele_key_dict[allele_key] = {}
                allele_key_dict[allele_key]['fraction_tags'] = fraction_tags
            n_alleles = len(allele_key_dict)
            ## Finds sequences that are probably just noisy versions of the same allele, by conditional probability of observing the same sequence within the same tagmentation site
            if n_alleles > 1:
                conditional_probs = compare_allele_stats(info_dict, allele_key_dict)
                for allele_key in allele_key_dict.keys():
                    allele_key_dict[allele_key]['conditional_probs'] = conditional_probs[allele_key]
            if info_dict['cell_w_tag_site'] > 200:
                allele_calls[ins_key] = allele_key_dict
        _save_pickle(allele_calls_path, allele_calls)

    ## Subset the insertions to only include those with valid allele sequence summaries.
    insertions_with_alleles = [k for k, v in allele_calls.items() if len(v) > 1]

    ##transformation_res = Parallel(n_jobs=threads, verbose = 3)(delayed(find_transformations)(bamfile, cell_BC_set, insertion_information[key], allele_calls[key], key) for key in insertions_with_alleles)
    ##transformation_dict = {ins_key: trans_dict for ins_key, trans_dict in transformation_res}
    ##with open('{}_transformation_dict.pkl'.format(prefix), 'wb') as handle:
    ##    pickle.dump(transformation_dict,handle,  protocol=pickle.HIGHEST_PROTOCOL)

    ## Stage 3: find sequencing-error transformations, apply corrections, and build histograms.
    if _is_cached(histogram_path, overwrite):
        _warn_skip(histogram_path)
        histogram_dfs_res_dict = _load_pickle(histogram_path)
    else:
        ##Find out which common sequencing errors exists per allele and (hopefully) how to correct them.
        corrections_to_find = {}
        for ins_key, allele_call in allele_calls.items():
            print(ins_key)
            corrections_to_find[ins_key] = {}
            for allele in allele_call.keys():
                corrections_to_find[ins_key][allele] = []
                noise_dict = res_dict[ins_key]['allele_count_k'][allele]
                for noise_allele, noise_count in noise_dict.items():
                    # If there are more than 200 reads coming from this noisy version, try to find a transformation to the true allele.
                    if noise_count > 200 and noise_allele != allele:
                        corrections_to_find[ins_key][allele].append(noise_allele)

        transformation_errors_res = Parallel(n_jobs=threads, verbose = 3)(delayed(find_transformation_errors_per_allele)(bamfile, cell_BC_set, insertion_information[key], corrections_to_find[key], key, min_len, break_len) for key in corrections_to_find.keys())
        transformation_corrections_dict = {ins_key: transformation_correction for ins_key,transformation_correction in zip(corrections_to_find.keys(),transformation_errors_res)}

        ## Figure out how to correct the tract lengths for common errors.
        count_correction_dict = {}
        for ins_key, transformation_corrections in transformation_corrections_dict.items():
            count_correction_dict[ins_key] = {}
            for allele_key, transformations in transformation_corrections.items():
                count_correction_dict[ins_key][allele_key] = {}
                for correction_seq, action_dict in transformations[1].items():
                    if len(action_dict) == 0:
                        continue
                    count_correction_dict[ins_key][allele_key][correction_seq] = correction_to_apply(action_dict[allele_key], allele_key)

        ## Make histograms with the corrections applied.
        histogram_dfs_res = Parallel(n_jobs=threads, verbose = 3)(delayed(get_histogram_dfs)(bamfile, cell_BC_set, insertion_information[ins_key], list(allele_calls[ins_key].keys()), count_correction_dict[ins_key], allele_calls[ins_key], ins_key, args.threshold, min_len, break_len) for ins_key in allele_calls.keys())
        histogram_dfs_res_dict = {k:v for k,v in histogram_dfs_res}
        _save_pickle(histogram_path, histogram_dfs_res_dict)

    ## Stage 4: scan for biallelic signal in insertions where only one allele sequence was detected.
    if _is_cached(peak_based_path, overwrite):
        _warn_skip(peak_based_path)
        peak_based_calling_result_dict = _load_pickle(peak_based_path)
    else:
        peak_based_calling_result_dict = {}
        for ins_key, test_peaks in histogram_dfs_res_dict.items():
            if len(allele_calls[ins_key]) != 1:
                continue
            biallele_result = scan_for_biallelic_signal(test_peaks)
            if len(biallele_result[0]) > 0:
                peak_based_calling_result_dict[ins_key] = biallele_result
        _save_pickle(peak_based_path, peak_based_calling_result_dict)

    ## Heterozygous SNP sites are needed for stages 5-7; only compute them if at least one is going to run.
    snp_cpm_cached = _is_cached(snp_cpm_path, overwrite)
    phase_allele_cached = _is_cached(phase_allele_path, overwrite)
    phase_biallele_cached = _is_cached(phase_biallele_path, overwrite)

    het_sites_dict = None
    insertions_with_het_sites = set()
    if not (snp_cpm_cached and phase_allele_cached and phase_biallele_cached):
        het_sites_res = Parallel(n_jobs=threads, verbose=3)(
            delayed(get_heterozygous_positions)(bamfile, insertion_information[key], key)
            for key in allele_calls.keys()
        )
        het_sites_dict = {k: v for k, v in het_sites_res}
        insertions_with_het_sites = set(k for k, v in het_sites_dict.items() if len(v) > 0)

    ## Stage 5: SNP calling using L1 flanking sequence CPM.
    if snp_cpm_cached:
        _warn_skip(snp_cpm_path)
        snp_cpm_dict = _load_pickle(snp_cpm_path)
    else:
        snp_cpm_res = Parallel(n_jobs=threads, verbose=3)(
            delayed(get_L1_flank_to_snp_cpm)(bamfile, cell_BC_set, insertion_information[key], allele_calls[key], het_sites_dict[key], key, min_len, break_len)
            for key in insertions_with_het_sites
        )
        snp_cpm_dict = {r[0]: r[1:] for r in snp_cpm_res}
        _save_pickle(snp_cpm_path, snp_cpm_dict)

    ## Stage 6: phase multi-allele sequence calling with SNV calling.
    if phase_allele_cached:
        _warn_skip(phase_allele_path)
        phase_allele_snv_dict = _load_pickle(phase_allele_path)
    else:
        insertions_with_alleles_and_het = [k for k in insertions_with_alleles if k in insertions_with_het_sites]
        phase_allele_snv_res = Parallel(n_jobs=threads, verbose=3)(
            delayed(phase_L1_and_flank_SNV)(bamfile, cell_BC_set, insertion_information[key], allele_calls[key], het_sites_dict[key], key, min_len, break_len)
            for key in insertions_with_alleles_and_het
        )
        phase_allele_snv_dict = {k: v for k, v in phase_allele_snv_res}
        _save_pickle(phase_allele_path, phase_allele_snv_dict)

    ## Stage 7: phase peak-based biallelic signal with SNV calling.
    if phase_biallele_cached:
        _warn_skip(phase_biallele_path)
        phase_biallele_snv_dict = _load_pickle(phase_biallele_path)
    else:
        peak_based_with_het = [k for k in peak_based_calling_result_dict.keys() if k in insertions_with_het_sites]
        phase_biallele_snv_res = Parallel(n_jobs=threads, verbose=3)(
            delayed(phase_biallelic_signal_and_SNV)(bamfile, cell_BC_set, insertion_information[key], allele_calls[key], het_sites_dict[key], key, histogram_dfs_res_dict[key], peak_based_calling_result_dict[key], min_len, break_len)
            for key in peak_based_with_het
        )
        phase_biallele_snv_dict = {k: v for k, v in phase_biallele_snv_res}
        _save_pickle(phase_biallele_path, phase_biallele_snv_dict)

    ## Stage 8: write per-insertion summary of all pipeline steps.
    if _is_cached(summary_path, overwrite):
        _warn_skip(summary_path)
    else:
        # build_insertion_summary needs het_sites_dict; load it if all SNV stages were cached.
        if het_sites_dict is None:
            het_sites_dict = {}
        summary_df = build_insertion_summary(
            insertion_information, res_dict, allele_calls,
            histogram_dfs_res_dict, peak_based_calling_result_dict,
            het_sites_dict, snp_cpm_dict,
            phase_allele_snv_dict, phase_biallele_snv_dict,
        )
        summary_df.to_csv(summary_path)


if __name__ == '__main__':
    main()

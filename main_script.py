from support_functions import *
import argparse
import json
from joblib import delayed,Parallel
import pandas as pd
import pickle
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
    args = parser.parse_args()

    bamfile = args.input
    threads = args.threads
    samplesheet_file = args.samplesheet
    prefix = args.prefix
    min_len = args.min_len
    break_len = args.break_len
    
    insertion_information = pd.read_csv(args.insertions, index_col = 0).T.to_dict()
    samplesheet = pd.read_csv(samplesheet_file, index_col=0)
    samplesheet_DNA = samplesheet[(samplesheet['Modality'] == 'DNA')]
    cell_BC_set = set(samplesheet_DNA.index)
    
    res = Parallel(n_jobs=threads, verbose = 3)(delayed(get_alleles_by_tagmentation_site)(bamfile, cell_BC_set, insertion_dict,key, min_len, break_len) for key, insertion_dict in insertion_information.items())

    res_dict = {t[0]: t[1] for t in res}

    with open('{}_res_dict.pkl'.format(prefix), 'wb') as handle:
        pickle.dump(res_dict,handle,  protocol=pickle.HIGHEST_PROTOCOL)
    
    ## Make a final decision on which sequence summaries represent allele variants.
    allele_calls = {}
    for ins_key, info_dict in res_dict.items():
        allele_counter = info_dict['allele_counter']
        #print(ins_key)
        tag_site_sum = sum(allele_counter.values())
        allele_key_dict = {}
        for allele_key, count in allele_counter.items():
            fraction_tags = count/tag_site_sum
            if fraction_tags < 0.1:
                continue
        #    print(allele_key, fraction_tags)
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
    
    with open('{}_allele_calls.pkl'.format(prefix), 'wb') as handle:
        pickle.dump(allele_calls,handle,  protocol=pickle.HIGHEST_PROTOCOL)

    ## Subset the insertions to only include those with valid allele sequence summaries.
    insertions_with_alleles = []
    for ins_key, ins_dict in allele_calls.items():
        if len(ins_dict.keys()) > 1:
            insertions_with_alleles.append(ins_key)
    
    
    ##transformation_res = Parallel(n_jobs=threads, verbose = 3)(delayed(find_transformations)(bamfile, cell_BC_set, insertion_information[key], allele_calls[key], key) for key in insertions_with_alleles)
    ##transformation_dict = {ins_key: trans_dict for ins_key, trans_dict in transformation_res}
    ##with open('{}_transformation_dict.pkl'.format(prefix), 'wb') as handle:
    ##    pickle.dump(transformation_dict,handle,  protocol=pickle.HIGHEST_PROTOCOL)
 
    ##Find out which common sequencing errors exists per allele and (hopefully) how to correct them.
    corrections_to_find = {}
    for ins_key, allele_call in allele_calls.items():
        print(ins_key)
        corrections_to_find[ins_key] = {}
        for allele, allele_dict in allele_call.items():
            corrections_to_find[ins_key][allele] = []
            #print('\t', allele, res_dict[ins_key]['allele_count_n'][allele])
            noise_dict = res_dict[ins_key]['allele_count_k'][allele]
            for noise_allele, noise_count in noise_dict.items():
                # If there are more than 200 reads coming from this noisy version, try to find a transformation to the true allele.
                if noise_count > 200 and noise_allele != allele:
                    corrections_to_find[ins_key][allele].append(noise_allele)
 
    transformation_errors_res = Parallel(n_jobs=threads, verbose = 3)(delayed(find_transformation_errors_per_allele)(bamfile, cell_BC_set, insertion_information[key], corrections_to_find[key], key) for key in corrections_to_find.keys())
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
    with open('{}_histogram_dfs.pkl'.format(prefix), 'wb') as handle:
        pickle.dump(histogram_dfs_res_dict, handle,  protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == '__main__':
    main()



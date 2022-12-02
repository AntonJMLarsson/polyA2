import pysam
import h5py
import pandas as pd
import numpy as np
import Levenshtein
from collections import Counter
import seqlogo
from itertools import combinations, permutations
import json
from joblib import delayed,Parallel
import seaborn as sns
import matplotlib.pyplot as plt

def get_quality_falloff(bamfile, cell_BC_set, insertion_dict, ins_key):
    bam = pysam.AlignmentFile(bamfile, 'rb')
    N = 0
    qual_mean = {p: 0 for p in range(127)}
    for read in bam.fetch(insertion_dict['chrom'], insertion_dict['start'], insertion_dict['end']):
        cell = read.get_tag('BC')
        if cell in cell_BC_set:
            if correct_orientation(read, insertion_dict['strand']):
                
                qual = read.get_tag('LQ')[23:]
                qual_list = list(qual)
                qual_list = [ord(c) - 33 for c in qual_list]
                for p, q in enumerate(qual_list):
                    qual_mean[p] = float((qual_mean[p] * N) + q) / (N + 1)
                N += 1
    return ins_key, qual_mean
def get_heterozygous_positions(bamfile, ins_info, key):
    bam = pysam.AlignmentFile(bamfile, 'rb')
    ins_res = []
    for pileupcolumn in bam.pileup(ins_info['chrom'], ins_info['start'], ins_info['end'] ):
        l = pileupcolumn.get_query_sequences()
        l = [c.upper() for c in l]
        if len(l) > 7000:
            n_bases = Counter(l)
            fraction_bases = Counter({c:count/sum(n_bases.values()) for c, count in n_bases.items()})
            most_common_fraction = fraction_bases.most_common()
            if most_common_fraction[0][1] < 0.6:
                a1 = most_common_fraction[0][0]
                a2 = most_common_fraction[1][0]
                a1_frac = most_common_fraction[0][1]
                a2_frac = most_common_fraction[1][1]
                if a1 != '' and a2 != '':
                    ins_res.append([pileupcolumn.reference_pos, a1, a2, a1_frac, a2_frac])
    return key, ins_res

def cell_contingency_table(ins_key, ins_dict, insertion_dict, bamfile, cell_BC_set):
    tagmentation_site_per_allele_counter = collect_tag_site_info(bamfile, cell_BC_set, insertion_dict)
    cell_detection_per_allele = {allele_key: set() for allele_key in ins_dict.keys()}
    
    for cell, tag_site_dicts in tagmentation_site_per_allele_counter.items():
        for p, tag_site_dict in tag_site_dicts.items():
            most_common_list = tag_site_dict.most_common()
            count = most_common_list[0][1]
            allele_key = most_common_list[0][0]
            if count > 10 and allele_key in cell_detection_per_allele:
                cell_detection_per_allele[allele_key] = cell_detection_per_allele[allele_key].union([cell])
    return cell_detection_per_allele
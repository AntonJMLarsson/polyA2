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

## used by collect_tag_site_info and get_sequence_pwm
def summarize_sequence(seq, cutoff, min_len, break_len):
    pre_string = ''
    post_string = ''
    A_counter = 0
    not_A_counter = 0
    contiguous_A_counter = 0
    pre_string_list = []
    A_tract_state = False
    break_state = False
    for b in seq:
        if b == 'A':
            contiguous_A_counter += 1
            # If we have previously found a non A base, determine if we should count it as the same tract.
            if break_state:
                post_string += b
                # If the previous two bases were As (and less than break_len non-As were found previously), count it as the same tract.
                if post_string[-(break_len+1):].count("A") == break_len+1:
                    post_string = ''
                    break_state = False
                    A_counter += break_len + 2
            else:
                A_counter += 1
                pre_string += b
            # here we can decide to enter the "A tract state" based on min_len. If we do, save the bases that precede the A tract.
            if A_counter >= min_len and not A_tract_state:
                not_A_counter = 0
                A_tract_state = True
                pre_string = pre_string[:-min_len].rstrip('A')
                pre_string_list.append(pre_string)
            # reset the not_A_counter for every min_len set of straight As in a row. To make sure we can skip multiple non-As that are sufficiently far apart.
            elif contiguous_A_counter >= min_len:
                not_A_counter = 0 
        if b != 'A':
            # if we are in a tract, determine if we should skip this base.
            if A_tract_state:
                break_state = True
                post_string += b
                not_A_counter += 1
                # If we have more than break_len non-As, we are done with this tract and should move on.
                if not_A_counter > break_len:
                    pre_string = post_string
                    post_string = ''
                    break_state = False
                    A_tract_state = False
                    A_counter = 0
            else:
                pre_string += b
                A_counter = 0
            contiguous_A_counter = 0
    # figure out what sequence is flanking our tract.
    pre_string_list.append(pre_string[:cutoff])
    return pre_string_list

def summarize_sequence_and_count_A_tracts(seq, cutoff, min_len, break_len):
    pre_string = ''
    post_string = ''
    A_counter = 0
    not_A_counter = 0
    contiguous_A_counter = 0
    pre_string_list = []
    A_count_list = []
    A_tract_state = False
    break_state = False
    for b in seq:
        if b == 'A':
            contiguous_A_counter += 1
            # If we have previously found a non A base, determine if we should count it as the same tract.
            if break_state:
                post_string += b
                # If the previous two bases were As (and less than break_len non-As were found previously), count it as the same tract.
                if post_string[-(break_len+1):].count("A") == break_len+1:
                    post_string = ''
                    break_state = False
                    A_counter += break_len + 2
            else:
                A_counter += 1
                pre_string += b
            # here we can decide to enter the "A tract state" based on min_len. If we do, save the bases that precede the A tract.
            if A_counter >= min_len and not A_tract_state:
                not_A_counter = 0
                A_tract_state = True
                pre_string = pre_string[:-min_len].rstrip('A')
                pre_string_list.append(pre_string)
            # reset the not_A_counter for every min_len set of straight As in a row. To make sure we can skip multiple non-As that are sufficiently far apart.
            elif contiguous_A_counter >= min_len:
                not_A_counter = 0 
        if b != 'A':
            # if we are in a tract, determine if we should skip this base.
            if A_tract_state:
                break_state = True
                post_string += b
                not_A_counter += 1
                # If we have more than break_len non-As, we are done with this tract and should move on.
                if not_A_counter > break_len:
                    pre_string = post_string
                    post_string = ''
                    break_state = False
                    A_tract_state = False
                    A_count_list.append(A_counter)
                    A_counter = 0
            else:
                pre_string += b
                A_counter = 0
            contiguous_A_counter = 0
    # figure out what sequence is flanking our tract.
    pre_string_list.append(pre_string[:cutoff])
    A_count_list.append(A_counter)
    return pre_string_list, A_count_list

def find_cells_with_sequence(read_dict, seq):
    cell_list = []
    for cell, seq_list in read_dict.items():
        if seq in seq_list:
            cell_list.append(cell)
    return cell_list


def count_A_tracts(seq):
    A_counter = 0
    A_count_list = []
    for b in seq:
        if b == 'A':
            A_counter += 1
        if b != 'A':
            if A_counter > 6:
                A_count_list.append(A_counter)
            A_counter = 0
    return A_count_list

def get_A_tract_length_for_allele(seqs, identical_reads, n_A_tracts):
    A_tract_lengths = {k: Counter() for k in range(n_A_tracts)}
    for seq in seqs:
        A_tract_tuple = count_A_tracts(seq)
        n = identical_reads[seq]
        for i, k in enumerate(A_tract_tuple):
            A_tract_lengths[i].update({k:n})
    return A_tract_lengths

def hamming_distance_unequal(s1, s2):
    min_length = min(len(s1), len(s2))
    return sum(ch1 != ch2 for ch1,ch2 in zip(s1[:min_length],s2[:min_length]))

def reconstruct_bottleneck_sequence(seq, flank_sequence, A_tract_lengths):
    reconstructed_sequence = ''
    seq_list = seq.split(',')
    for i, s in enumerate(seq_list[:-1]):
        s_new = s + A_tract_lengths[i]*'A'
        reconstructed_sequence += s_new
    reconstructed_sequence += flank_sequence
    return reconstructed_sequence

## used by collect_tag_site_info, ensure the read is in the correct orientation relative to the insertion (always opposite orientation).
def correct_orientation(read, strand):
    if (read.is_reverse and strand == '+') or (not read.is_reverse and strand == '-'):
        return True
    else:
        return False

def collect_tag_site_numbers(bamfile, cell_BC_set, insertion_dict, key):
    bam = pysam.AlignmentFile(bamfile,'rb')
    tagmentation_sites_per_cell = {cell: [] for cell in cell_BC_set}
    for read in bam.fetch(insertion_dict['chrom'],insertion_dict['start'],insertion_dict['end']):
        cell = read.get_tag('BC')
        if cell in cell_BC_set:
            if correct_orientation(read, insertion_dict['strand']):
                tagmentation_site = read.reference_end if read.is_reverse else read.reference_start
                tagmentation_sites_per_cell[cell].append(tagmentation_site)
            else:
                continue
    tagmentation_sites_counter = Counter()
    for cell, tag_list in tagmentation_sites_per_cell.items():
        tag_list_counter = Counter(tag_list)
        n_tag_sites = len([tag for tag, count in tag_list_counter.items() if count > 10])
        tagmentation_sites_counter.update({n_tag_sites:1})
    
    return key, tagmentation_sites_counter

## used by get_alleles_by_tagmentation_site, collects the sequence summaries for each tagmentation site within each cell.
def collect_tag_site_info(bamfile, cell_BC_set, insertion_dict, min_len, break_len):
    bam = pysam.AlignmentFile(bamfile,'rb')
    tagmentation_site_per_allele = {cell: {} for cell in cell_BC_set}
    contigs = set([d['SN'] for d in bam.header['SQ']])
    if insertion_dict['chrom'] not in contigs:
        return tagmentation_site_per_allele
    for read in bam.fetch(insertion_dict['chrom'],insertion_dict['start'],insertion_dict['end']):
        cell = read.get_tag('BC')
        if cell in cell_BC_set:
            if correct_orientation(read, insertion_dict['strand']):
                seq = read.get_tag('LS')
                seq_summary = ','.join(summarize_sequence(seq,5, min_len, break_len))
                tagmentation_site = read.reference_end if read.is_reverse else read.reference_start
                if seq_summary in tagmentation_site_per_allele[cell]:
                    tagmentation_site_per_allele[cell][seq_summary].append(tagmentation_site)
                else:
                    tagmentation_site_per_allele[cell][seq_summary] = [tagmentation_site]
            else:
                continue
    tagmentation_site_per_allele_counter = {cell: {} for cell in cell_BC_set}
    for cell, allele_dict in tagmentation_site_per_allele.items():
        for k,l in allele_dict.items():
            c = Counter(l)
            for p, n in c.items():
                if p in tagmentation_site_per_allele_counter[cell]:
                    tagmentation_site_per_allele_counter[cell][p].update({k:n})
                else:
                    tagmentation_site_per_allele_counter[cell][p] = Counter({k:n})
    
    return tagmentation_site_per_allele_counter

## used by get_alleles_by_tagmentation_site. The main thing this function does is get candidate representations of alleles by taking the most common sequence summary
## per tagmentation sequence (and cell), if that sequence has a read count above 10.
def get_correct_alleles(tagmentation_site_per_allele_counter):
    allele_counter = Counter()
    tag_site_distribution_dict = {}
    count_distribution_dict = {}
    cell_w_counts = 0
    cell_w_tag_site = 0
    for cell, tag_site_dicts in tagmentation_site_per_allele_counter.items():
        has_counts = False
        has_tag_site = False
        for p, tag_site_dict in tag_site_dicts.items():
            if sum(tag_site_dict.values()) > 10:
                has_counts = True
            most_common_list = tag_site_dict.most_common()
            count = most_common_list[0][1]
            allele_key = most_common_list[0][0]
            if count > 10:
                has_tag_site = True
                allele_counter.update({allele_key:1})
                if allele_key in tag_site_distribution_dict:
                    tag_site_distribution_dict[allele_key].update({p:1})
                    count_distribution_dict[allele_key].update({count:1})
                else:
                    tag_site_distribution_dict[allele_key] = Counter({p:1})
                    count_distribution_dict[allele_key] = Counter({count:1})
        if has_counts:
            cell_w_counts += 1
        if has_tag_site:
            cell_w_tag_site += 1
    return allele_counter, tag_site_distribution_dict, count_distribution_dict, cell_w_counts, cell_w_tag_site

## used by get_alleles_by_tagmentation_site, finds the most common sequences in within a tagmentation site (per cell) to find the representative sequence for each allele.
def get_allele_coupling(tagmentation_site_per_allele_counter, allele_counter):
    allele_count_n = {allele_key: 0 for allele_key in allele_counter.keys()}
    allele_count_k = {allele_key: Counter({allele_key: 0 for allele_key in allele_counter.keys()}) for allele_key in allele_counter.keys()}
    allele_key_set = set(allele_counter.keys())
    for cell, tag_site_dicts in tagmentation_site_per_allele_counter.items():
        for p, tag_site_dict in tag_site_dicts.items():
            most_common_list = tag_site_dict.most_common()
            count = most_common_list[0][1]
            allele_key = most_common_list[0][0]
            other_allele_keys = allele_key_set.intersection(set(tag_site_dict.keys()))
            if count > 10:
                allele_count_n[allele_key] += count
                for other_key in other_allele_keys:
                    allele_count_k[allele_key].update({other_key: tag_site_dict[other_key]})
    return allele_count_n, allele_count_k

## used in main
def get_alleles_by_tagmentation_site(bamfile, cell_BC_set, insertion_dict, key, min_len, break_len):
    
    tagmentation_site_per_allele_counter = collect_tag_site_info(bamfile, cell_BC_set, insertion_dict, min_len, break_len)
    allele_counter, tag_site_distribution_dict, count_distribution_dict, cell_w_counts, cell_w_tag_site = get_correct_alleles(tagmentation_site_per_allele_counter)
    
    allele_count_n, allele_count_k = get_allele_coupling(tagmentation_site_per_allele_counter, allele_counter)
    
    result_dict = {'allele_counter':allele_counter, 'tag_site_distribution_dict':tag_site_distribution_dict, 
                   'count_distribution_dict': count_distribution_dict, 'cell_w_counts':cell_w_counts,
                   'cell_w_tag_site': cell_w_tag_site, 'allele_count_n':allele_count_n, 
                   'allele_count_k': allele_count_k}
    
    return key, result_dict

## used in main
def compare_allele_stats(info_dict, allele_key_dict):
    conditional_probs = {a1: {} for a1 in allele_key_dict.keys()}
    for a1, a2 in permutations(allele_key_dict.keys(),2):
        conditional_probs[a1][a2] = info_dict['allele_count_k'][a1][a2]/info_dict['allele_count_k'][a1][a1]
    return conditional_probs

def get_operation_list(destination_motif):
    summary_list = destination_motif.split(',')
    n = len(summary_list)
    operation_list = []
    for i, seq in enumerate(summary_list):
        if i == 0:
            seq_reverse = seq[::-1]
            op_seq = get_op_seq(seq_reverse)
            operation_list.append((seq, op_seq[::-1]))
        elif i != n-1:
            op_seq_begin = get_op_seq(seq)
            seq_reverse = seq[::-1]
            op_seq_end = get_op_seq(seq_reverse)
            operation_list.append((seq, (op_seq_begin, op_seq_end[::-1])))
        else:
            op_seq = get_op_seq(seq)
            operation_list.append((seq, op_seq))
    return operation_list

def hamming_distance_unequal(s1, s2):
    min_length = min(len(s1), len(s2))
    return sum(ch1 != ch2 for ch1,ch2 in zip(s1[:min_length],s2[:min_length]))

def determine_adding_action(source_motif_list, destination_motif_list, source_flank, destination_flank):
    # Find the sequence from the destination list to add to the source motif list and where
    
    for i, seq in enumerate(destination_motif_list):
        if len(seq) > 15:
            if hamming_distance_unequal(seq[:15], source_flank) < 2:
                action = ('truncate', i,0,'')
                return action
    
    lev_distance = {k: 0 for k in range(len(destination_motif_list))}
    for k in range(len(destination_motif_list)):
        temp_destination_motif_list = destination_motif_list[:k] + destination_motif_list[(k+1):]
        
        for seq1, seq2 in zip(temp_destination_motif_list, source_motif_list):
            lev_distance[k] += Levenshtein.distance(seq1, seq2)
    k = min(lev_distance, key=lev_distance.get)
    action = ('insert', k,k,destination_motif_list[k])
    return action

def determine_remove_action(source_motif_list, destination_motif_list, source_flank, destination_flank):
    # Find the best sequence from the source list to remove
    for i, seq in enumerate(source_motif_list):
        if len(seq) > 15:
            if hamming_distance_unequal(seq[:15], destination_flank) < 2:
                action = ('truncate', i,0,'')
                return action
    
    lev_distance = {k: 0 for k in range(len(source_motif_list))}
    for k in range(len(source_motif_list)):
        temp_source_motif_list = source_motif_list[:k] + source_motif_list[(k+1):]
        
        for seq1, seq2 in zip(temp_source_motif_list, destination_motif_list):
            lev_distance[k] += Levenshtein.distance(seq1, seq2)
    action = ('remove', min(lev_distance, key=lev_distance.get), 0, '')
    return action

def determine_increasing_action(source_seq, destination_seq, operations, source_flank, destination_flank, i, n):
    if i == 0:
        
        append_seq = operations[1]
        
        append_source_seq = source_seq + append_seq
        
        source_seq_lev = Levenshtein.distance(source_seq, destination_seq)
        append_source_seq_lev = Levenshtein.distance(append_source_seq, destination_seq)
        
        if source_seq_lev <= append_source_seq_lev:
            action = ('abort', 0,0,0)
        else:
            action = ('append', i, 0, append_seq)
        
    elif i == n:
        action = ('append', i, 0, destination_seq[len(source_seq):])
    else:
        prepend_seq = operations[1][0]
        append_seq = operations[1][1]
        
        prepend_source_seq = prepend_seq + source_seq
        append_source_seq = source_seq + append_seq
        
        source_seq_lev = Levenshtein.distance(source_seq, destination_seq)
        
        prepend_source_seq_lev = Levenshtein.distance(prepend_source_seq, destination_seq)
        
        append_source_seq_lev = Levenshtein.distance(append_source_seq, destination_seq)
        
        if source_seq_lev < prepend_source_seq_lev and source_seq_lev < append_source_seq_lev:
            action = ('abort', 0,0,0)
        elif prepend_source_seq_lev == append_source_seq_lev:
            if len(prepend_source_seq) > len(destination_seq):
                action = ('abort', 0,0,0)
            else:
                action = ('prepend', i, 'abort', prepend_seq)
        elif prepend_source_seq_lev < append_source_seq_lev:
            if len(prepend_source_seq) > len(destination_seq):
                action = ('abort', 0,0,0)
            else:
                action = ('prepend', i, 0, prepend_seq)
        elif append_source_seq_lev < prepend_source_seq_lev:
            if len(append_source_seq) > len(destination_seq):
                action = ('abort', 0,0,0)
            else:
                action = ('append', i, 0, append_seq)
    return action

def determine_decreasing_action(source_seq, destination_seq, source_operations,source_flank, destination_flank, i, n):
    if i == 0:
        
        rstrip_len = len(source_operations[1])
        
        rstrip_source_seq = source_seq[:-rstrip_len]
        
        source_seq_lev = Levenshtein.distance(source_seq, destination_seq)
        rstrip_source_seq_lev = Levenshtein.distance(rstrip_source_seq, destination_seq)
        
        if source_seq_lev <= rstrip_source_seq_lev:
            action = ('abort', 0,0,0)
        else:
            action = ('strip', i, rstrip_len, 'r')
    elif i == n:
        action = ('strip', i, len(source_seq)-len(destination_seq),'r')
    else:
        rstrip_len = len(source_operations[1][1])
        lstrip_len = len(source_operations[1][0])
        
        rstrip_source_seq = source_seq[:-rstrip_len]
        lstrip_source_seq = source_seq[lstrip_len:]
        
        source_seq_lev = Levenshtein.distance(source_seq, destination_seq)
        
        rstrip_source_seq_lev = Levenshtein.distance(rstrip_source_seq, destination_seq)
        
        lstrip_source_seq_lev = Levenshtein.distance(lstrip_source_seq, destination_seq)
        
        if source_seq_lev < lstrip_source_seq_lev and source_seq_lev < rstrip_source_seq_lev:
            action = ('abort', 0,0,0)
        elif lstrip_source_seq_lev == rstrip_source_seq_lev:
            if len(rstrip_source_seq) > len(destination_seq):
                action = ('abort', 0,0,'')
            else:
                action = ('strip', i, rstrip_len, 'abort')
        elif lstrip_source_seq_lev < rstrip_source_seq_lev :
            if len(lstrip_source_seq) > len(destination_seq):
                action = ('abort', 0,0,0)
            else:
                action = ('strip', i, lstrip_len, 'l')
        elif rstrip_source_seq_lev < lstrip_source_seq_lev :
            if len(rstrip_source_seq) > len(destination_seq):
                action = ('abort', 0,0,0)
            else:
                action = ('strip', i, rstrip_len, 'r')
    return action

def determing_editing_action(source_seq, destination_seq, destination_operations, source_operations, source_flank, destination_flank, i, n):
    if i == n:
        edit_dist = hamming_distance_unequal(source_flank, destination_flank)
        prepend_dist = hamming_distance_unequal(destination_operations[1]+source_flank, destination_flank)
        lstrip_dist = hamming_distance_unequal(source_flank[len(source_operations[1]):], destination_flank)
        
        if edit_dist == prepend_dist:
            action = ('abort', 0,0,0)
        elif edit_dist < prepend_dist and edit_dist < lstrip_dist:
            for j, (s_char, d_char) in enumerate(zip(source_seq, destination_seq)):
                if s_char != d_char:
                    action = ('edit', i, j, d_char)
                    return action
        elif prepend_dist == lstrip_dist:
            action = ('abort', 0, 0, 0)
        elif prepend_dist < lstrip_dist:
            action = ('prepend', i, 5, destination_operations[1])
        else:
            action = ('strip', i, len(source_operations[1]), 'l')
    else:
        for j, (s_char, d_char) in enumerate(zip(source_seq, destination_seq)):
            if s_char != d_char:
                action = ('edit', i, j, d_char)
                return action
    return action

def determine_next_action(source_motif_list, destination_motif_list, source_flank, destination_flank, destination_operation_list, i, n):
    # if the source motif is shorter than destination, we need to add a sequence. 
    # we'll start with insert, but split is TODO. If we insert, polyA information is lost.
    if len(source_motif_list) < len(destination_motif_list):
        action = determine_adding_action(source_motif_list, destination_motif_list, source_flank, destination_flank)
    # what to remove
    elif len(destination_motif_list) < len(source_motif_list):
        action = determine_remove_action(source_motif_list, destination_motif_list, source_flank, destination_flank)
    else:
        source_seq = source_motif_list[i]
        destination_seq = destination_motif_list[i]
        if source_seq == destination_seq:
            return None
        elif len(source_seq) < len(destination_seq):
            destination_operations = destination_operation_list[i]
            action = determine_increasing_action(source_seq, destination_seq, destination_operations, source_flank, destination_flank, i, n)
        elif len(destination_seq) < len(source_seq):
            source_operations = get_operation_list(','.join(source_motif_list))[i]
            action = determine_decreasing_action(source_seq, destination_seq, source_operations, source_flank, destination_flank, i, n)
        else:
            destination_operations = destination_operation_list[i]
            source_operations = get_operation_list(','.join(source_motif_list))[i]
            action = determing_editing_action(source_seq, destination_seq, destination_operations, source_operations, source_flank, destination_flank, i, n)
    return action

## used by collect_flanking_sequences
def group_identical_reads(read_dict):
    identical_reads = {}
    for cell, seq_list in read_dict.items():
        for seq in seq_list:
            if seq in identical_reads.keys():
                identical_reads[seq] += 1     
            else:
                identical_reads[seq] = 1
    return identical_reads

## used by collect_flanking_sequences
def group_sequences(identical_reads, min_len, break_len):
    sequence_summaries_count = Counter()
    sequence_summaries = {}
    for seq,n in identical_reads.items():
        s = ','.join(summarize_sequence(seq,5, min_len, break_len))
        if s in sequence_summaries:
            sequence_summaries[s].append(seq)
        else:
            sequence_summaries[s] = [seq]
        sequence_summaries_count.update({s:n})
    return sequence_summaries,sequence_summaries_count

## used by collect_flanking_sequences
def get_sequence_pwm(seqs, identical_reads, min_len, break_len):
    base_freqs = Counter()
    for seq in seqs:
        last_sequence = summarize_sequence(seq,1000, min_len, break_len)[-1]
        c = Counter(enumerate(last_sequence))
        n = identical_reads[seq]
        for k in c.keys():
            c[k] = n
        base_freqs.update(c)
    return base_freqs

## used by find_transformations
def collect_flanking_sequences(bamfile, cell_BC_set, insertion_dict, allele_list, min_len, break_len):
    bam = pysam.AlignmentFile(bamfile, 'r')

    read_dict = {cell: [] for cell in cell_BC_set}
    for read in bam.fetch(insertion_dict['chrom'],insertion_dict['start'],insertion_dict['end']):
        cell = read.get_tag('BC')
        if cell in cell_BC_set:
            if correct_orientation(read, insertion_dict['strand']):
                LS = read.get_tag('LS')
                read_dict[cell].append(LS)
            else:
                continue
    read_dict = {cell: l for cell, l in read_dict.items() if len(l) > 0}

    identical_reads = group_identical_reads(read_dict)
    sequence_summaries,sequence_summaries_count = group_sequences(identical_reads, min_len, break_len)
    
    cpm_dict = {}
    for allele_key in allele_list:
        d = get_sequence_pwm(sequence_summaries[allele_key], identical_reads, min_len, break_len)
        pfm = pd.Series(d).unstack(fill_value=0).drop('N', errors='ignore',axis=1)
        for b in ['A','T', 'C', 'G']:
            if b not in pfm.columns:
                pfm[b] = 0
        cpm = seqlogo.CompletePm(pfm)
        cpm_dict[allele_key] = cpm
    return cpm_dict

## skipped for now in main
def find_transformations(bamfile, cell_BC_set, insertion_dict, allele_call, key, min_len, break_len):
    ## Find out how the tract is flanked.
    cpm_dict = collect_flanking_sequences(bamfile, cell_BC_set, insertion_dict, allele_call, min_len, break_len)
    flanking_dict = {allele_key: cpm.consensus for allele_key, cpm in cpm_dict.items()}
    operations_dict = {k: {} for k in flanking_dict.keys()}
    for k1, k2 in permutations(flanking_dict, 2):
        operations_dict[k1][k2] = calculate_minimum_operations(k1, flanking_dict[k1], k2, flanking_dict[k2])
    return key, operations_dict


def find_transformation_errors_per_allele(bamfile, cell_BC_set, insertion_dict, noise_dict, key, min_len, break_len):
    res_dict = {}
    for origin_allele, noise_alleles in noise_dict.items():
        res_dict[origin_allele] = find_transformation_errors(bamfile, cell_BC_set, insertion_dict, origin_allele, noise_alleles, key, min_len, break_len)
    return res_dict


def find_transformation_errors(bamfile, cell_BC_set, insertion_dict, origin_allele, noise_alleles, key, min_len, break_len):
    all_alleles = noise_alleles.copy()
    all_alleles.append(origin_allele)
    cpm_dict = collect_flanking_sequences(bamfile, cell_BC_set, insertion_dict, all_alleles, min_len, break_len)
    flanking_dict = {allele_key: cpm.consensus for allele_key, cpm in cpm_dict.items()}
    operations_dict = {k: {} for k in flanking_dict.keys()}
    for k1 in noise_alleles:
        operations_dict[k1][origin_allele] = calculate_minimum_operations(k1, flanking_dict[k1], origin_allele, flanking_dict[origin_allele])
    return key, operations_dict

## used in main, tries to figure out how to edit a sequence summary (and the resulting tract lengths) to obtain the correct histogram.
def correction_to_apply(action_list, destination_allele):
    correction_dict = {i: 0 for i in range(destination_allele.count(','))}
    abort_set = set(['insert', 'abort', 'infinite', 'remove'])
    if any([any([c in abort_set for c in action]) for action in action_list]):
        return None
                
    for action in action_list:
        if action[0] == 'strip':
            if action[3] == 'r':
                if action[1] == len(correction_dict):
                    continue
                correction_dict[action[1]] += action[2]
            elif action[3] == 'l':
                correction_dict[action[1]-1] += action[2]
        elif action[0] == 'prepend':
            correction_dict[action[1]-1] -= len(action[3])
        elif action[0] == 'append':
            if action[1] == len(correction_dict):
                continue
            correction_dict[action[1]] -= len(action[3])
    return correction_dict
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
def get_allele_to_snp_counts(bamfile, cell_BC_set, insertion_dict, allele_call, het_site, ins_key, min_len, break_len):
    bam = pysam.AlignmentFile(bamfile, 'rb')
    allele_to_snp = {allele_summary : Counter({het_site[1]: 0, het_site[2]: 0}) for allele_summary in allele_call.keys()}
    for read in bam.fetch(insertion_dict['chrom'], het_site[0]-1, het_site[0]+1):
        cell = read.get_tag('BC')
        if cell in cell_BC_set:
            if correct_orientation(read, insertion_dict['strand']):
                seq = read.get_tag('LS')
                seq_summary = ','.join(summarize_sequence(seq,5, min_len, break_len))
                if seq_summary in allele_to_snp:
                    pos_list = read.get_reference_positions()
                    seq_list = read.query_alignment_sequence
                    qual_list = read.query_alignment_qualities
                    if het_site[0] in pos_list:
                        idx = pos_list.index(het_site[0])
                        b = seq_list[idx]
                        q = qual_list[idx]
                        if b in het_site and q > 20:
                            allele_to_snp[seq_summary].update({b:1})
    return allele_to_snp
def get_CPMs_for_snp(bamfile, cell_BC_set, insertion_dict, het_site, allele_summary, min_len, break_len):
    bam = pysam.AlignmentFile(bamfile, 'r')
    
    flank_dicts = {b: Counter() for b in [het_site[1], het_site[2]]}
    flank_size = {b: Counter()  for b in [het_site[1], het_site[2]]}
    for read in bam.fetch(insertion_dict['chrom'], het_site[0]-1, het_site[0]+1):
        cell = read.get_tag('BC')
        if cell in cell_BC_set:
            if correct_orientation(read, insertion_dict['strand']):
                pos_list = read.get_reference_positions()
                seq_list = read.query_alignment_sequence
                qual_list = read.query_alignment_qualities
                if het_site[0] in pos_list:
                    seq = read.get_tag('LS')
                    summary_list = summarize_sequence(seq,1000, min_len, break_len)
                    summary_list_small = summary_list.copy()
                    summary_list_small[-1] = summary_list_small[-1][:5]
                    seq_summary_small = ','.join(summary_list_small)
                    if seq_summary_small == allele_summary:
                        idx = pos_list.index(het_site[0])
                        b = seq_list[idx]
                        q = qual_list[idx]
                        if b in flank_dicts and q > 20:
                            flank_seq = summary_list[-1]
                            flank_dicts[b].update(enumerate(flank_seq))
                            flank_size[b].update({len(flank_seq): 1})
            else:
                continue
    
    cpm_dicts = {}
    for SNV_b, count_d in flank_dicts.items():
        try:
            pfm = pd.Series(count_d).unstack(fill_value=0).drop('N', errors='ignore',axis=1)
        except ValueError:
            return None, flank_size
        for b in ['A','T', 'C', 'G']:
            if b not in pfm.columns:
                pfm[b] = 0
        cpm = seqlogo.CompletePm(pfm)
        cpm_dicts[SNV_b] = cpm
    return cpm_dicts, flank_size
def get_fraction_less(p, count_dict):
    total = sum(count_dict.values())
    less = 0
    for p2, count in count_dict.items():
        if p < p2:
            less += count
            
    return less/total

def get_L1_flank_to_snp_cpm(bamfile, cell_BC_set, insertion_dict, allele_call, het_sites, ins_key, min_len, break_len):
    output_dict = {}
    allele_summary = list(allele_call.keys())[0]
    for het_site in het_sites:
        cpm_dicts, flank_size = get_CPMs_for_snp(bamfile, cell_BC_set, insertion_dict, het_site, allele_summary, min_len, break_len)
        if cpm_dicts is None:
            return ins_key, None, None
        output_dict[het_site[0]] = (cpm_dicts, flank_size)
    
    variants_phased = {}
    unequal_len = False
    for p, (cpm_dicts, flank_size) in output_dict.items():
        variants = list(cpm_dicts.keys())
        cpm_a1 = cpm_dicts[variants[0]]
        cpm_a2 = cpm_dicts[variants[1]]
        
        a1_len = cpm_a1.ppm.shape[0]
        a2_len = cpm_a2.ppm.shape[0]
        
        if a1_len != a2_len:
            unequal_len = True
            len_min = min(a1_len, a2_len)
            cpm_a1_ppm = cpm_a1.ppm.loc[range(0,len_min)]
            cpm_a2_ppm = cpm_a2.ppm.loc[range(0,len_min)]
            cpm_a1_pwm = cpm_a1.pwm.loc[range(0,len_min)]
            cpm_a2_pwm = cpm_a2.pwm.loc[range(0,len_min)]
        else:
            cpm_a1_ppm = cpm_a1.ppm
            cpm_a2_ppm = cpm_a2.ppm
            cpm_a1_pwm = cpm_a1.pwm
            cpm_a2_pwm = cpm_a2.pwm
        
        variants_phased[p] = {}
        for pos in cpm_a1_ppm.index[(cpm_a1_ppm*(cpm_a1_pwm - cpm_a2_pwm)).sum(axis=1) > 6].values:
            a1 = cpm_a1_ppm.loc[pos].idxmax()
            a2 = cpm_a2_ppm.loc[pos].idxmax()
            if a1 == a2 or cpm_a1_ppm.loc[pos][a1] < 0.95 or cpm_a2_ppm.loc[pos][a2] < 0.95:
                continue
            if min([get_fraction_less(pos, flank_size[v]) for v in variants]) > 0.98:
                variants_phased[p][pos] = {variants[0]: a1, variants[1]: a2}
    return ins_key, variants_phased, unequal_len, output_dict

from scipy.stats import chi2_contingency
def phase_L1_and_flank_SNV(bamfile, cell_BC_set, insertion_dict, allele_call, het_sites, ins_key, min_len, break_len):
    output_dict = {}
    for het_site in het_sites:
        allele_to_snp = get_allele_to_snp_counts(bamfile, cell_BC_set, insertion_dict, allele_call, het_site, ins_key, min_len, break_len)
        chi2, p , dof, expected = chi2_contingency([[c for c in variant_counts.values()] for variant_counts in allele_to_snp.values()])
        output_dict[het_site[0]] = (allele_to_snp, chi2, p , dof, expected)
    return ins_key, output_dict

def get_biallele_to_snp_counts(bamfile, cell_BC_set, insertion_dict, allele_call, het_site, test_peaks, boundaries_dict, min_len, break_len):
    allele_summary = list(allele_call.keys())[0]

    test_peaks_reset = test_peaks.reset_index()
    cell_tag_assignments = {}
    for _, row in test_peaks_reset.iterrows():
        t_key = row['t_key']
        if t_key not in boundaries_dict:
            continue
        hist_peak = row['hist_peak']
        if np.isnan(hist_peak):
            continue
        hist_peak_int = int(hist_peak)
        if hist_peak_int not in boundaries_dict[t_key]['biallelic_boundaries']:
            continue
        allele_idx = boundaries_dict[t_key]['biallelic_boundaries'][hist_peak_int]
        k = (row['cell'], row['tag_site'])
        if k not in cell_tag_assignments:
            cell_tag_assignments[k] = set()
        cell_tag_assignments[k].add(allele_idx)

    cell_tag_to_biallele = {k: list(v)[0] for k, v in cell_tag_assignments.items() if len(v) == 1}

    biallele_to_snp = {0: Counter({het_site[1]: 0, het_site[2]: 0}),
                       1: Counter({het_site[1]: 0, het_site[2]: 0})}

    bam = pysam.AlignmentFile(bamfile, 'rb')
    for read in bam.fetch(insertion_dict['chrom'], het_site[0]-1, het_site[0]+1):
        cell = read.get_tag('BC')
        if cell not in cell_BC_set:
            continue
        if not correct_orientation(read, insertion_dict['strand']):
            continue
        seq = read.get_tag('LS')
        seq_summary = ','.join(summarize_sequence(seq, 5, min_len, break_len))
        if seq_summary != allele_summary:
            continue
        tag_site = read.reference_end if read.is_reverse else read.reference_start
        if (cell, tag_site) not in cell_tag_to_biallele:
            continue
        allele_idx = cell_tag_to_biallele[(cell, tag_site)]
        pos_list = read.get_reference_positions()
        seq_list = read.query_alignment_sequence
        qual_list = read.query_alignment_qualities
        if het_site[0] in pos_list:
            idx = pos_list.index(het_site[0])
            b = seq_list[idx]
            q = qual_list[idx]
            if b in [het_site[1], het_site[2]] and q > 20:
                biallele_to_snp[allele_idx].update({b: 1})
    return biallele_to_snp

def phase_biallelic_signal_and_SNV(bamfile, cell_BC_set, insertion_dict, allele_call, het_sites, ins_key, test_peaks, biallele_result, min_len, break_len):
    boundaries_dict, _, _ = biallele_result
    output_dict = {}
    for het_site in het_sites:
        biallele_to_snp = get_biallele_to_snp_counts(bamfile, cell_BC_set, insertion_dict, allele_call, het_site, test_peaks, boundaries_dict, min_len, break_len)
        contingency = [[c for c in variant_counts.values()] for variant_counts in biallele_to_snp.values()]
        try:
            chi2, p, dof, expected = chi2_contingency(contingency)
        except ValueError:
            chi2, p, dof, expected = None, None, None, None
        output_dict[het_site[0]] = (biallele_to_snp, chi2, p, dof, expected)
    return ins_key, output_dict

def get_A_counter_dict(bamfile, cell_BC_set, insertion_dict, allele_call, ins_key, threshold, min_len, break_len):
    bam = pysam.AlignmentFile(bamfile, 'rb')
    seq_to_count_dict = {}
    allele_summary_set = set(list(allele_call.keys()))
    allele_summary_to_id = {a: i for i,a in enumerate(allele_call.keys())}
    A_count_dict = {cell: {a: {} for a in allele_call.keys()} for cell in cell_BC_set}
    for read in bam.fetch(insertion_dict['chrom'], insertion_dict['start'], insertion_dict['end']):
        cell = read.get_tag('BC')
        if cell not in cell_BC_set:
            continue
        if not correct_orientation(read, insertion_dict['strand']):
            continue
        seq = read.get_tag('LS')
        if seq in seq_to_count_dict:
            seq_info = seq_to_count_dict[seq]
            tagmentation_site = read.reference_end if read.is_reverse else read.reference_start
            if tagmentation_site in A_count_dict[cell][seq_info[0]]:
                A_count_dict[cell][seq_info[0]][tagmentation_site].append(seq_info[1])
            else:
                A_count_dict[cell][seq_info[0]][tagmentation_site] = [seq_info[1]]
        else:
            pre_string_list, A_count_list = summarize_sequence_and_count_A_tracts(seq, 5, min_len, break_len)
            seq_summary = ','.join(pre_string_list)
            if seq_summary not in allele_summary_set:
                continue
            tagmentation_site = read.reference_end if read.is_reverse else read.reference_start
            if tagmentation_site in A_count_dict[cell][seq_summary]:
                A_count_dict[cell][seq_summary][tagmentation_site].append(A_count_list)
            else:
                A_count_dict[cell][seq_summary][tagmentation_site] = [A_count_list]
            seq_to_count_dict[seq] = (seq_summary, A_count_list)

    A_counter_dict = {cell: {allele_summary_to_id[a]: {} for a in allele_call.keys()} for cell in cell_BC_set}
    
    A_count_len_dict = {cell: {} for cell in cell_BC_set}
    
    for cell, allele_dict in A_count_dict.items():
        for a, tag_site_dict in allele_dict.items():
            for p, A_count_list in tag_site_dict.items():
                if p in A_count_len_dict[cell]:
                    A_count_len_dict[cell][p][a] = len(A_count_list)
                else:
                    A_count_len_dict[cell][p] = {a: len(A_count_list)}
    
    A_count_allow_list = {cell: {} for cell in cell_BC_set}
    
    for cell, position_dict in A_count_len_dict.items():
        for p, a_dict in position_dict.items():
            if len(a_dict) == 1:
                continue
            max_allele = max(a_dict, key=a_dict.get)
            
            A_count_allow_list[cell][p] = [allele_summary_to_id[max_allele]]
            for a, a_count in a_dict.items():
                if a == max_allele:
                    continue
                cond_prob_val = allele_call[a]['conditional_probs'][max_allele]
                if cond_prob_val*a_dict[max_allele]*20 < a_count:
                    A_count_allow_list[cell][p].append(allele_summary_to_id[a])
    
    for cell, allele_dict in A_count_dict.items():
        for a, tag_site_dict in allele_dict.items():
            a_idx = allele_summary_to_id[a]
            for p, A_count_list in tag_site_dict.items():
                if len(A_count_list) < threshold:
                    continue
                if p in A_count_allow_list[cell]:
                    if a_idx not in A_count_allow_list[cell][p]:
                        continue
                A_counter_dict[cell][a_idx][p] = {}
                for i in range(a.count(',')):
                    A_counter_dict[cell][a_idx][p][i] = Counter([l[i] for l in A_count_list])
    
    
    return ins_key, A_counter_dict

def get_A_counter_dict_w_corrections(bamfile, cell_BC_set, insertion_dict, allele_list, seq_to_correct_per_allele, allele_call, ins_key, threshold, min_len, break_len):
    bam = pysam.AlignmentFile(bamfile, 'rb')
    seq_to_count_dict = {}
    allele_summary_set = set(allele_list)
    seq_summaries_to_correct = set()
    for seq_to_correct in seq_to_correct_per_allele.values():
        seq_summaries_to_correct = seq_summaries_to_correct.union(set(seq_to_correct.keys()))
    all_seq_summaries = allele_summary_set.union(seq_summaries_to_correct)
    allele_summary_to_id = {a: i for i,a in enumerate(allele_list)}
    A_count_dict = {cell: {a: {} for a in all_seq_summaries} for cell in cell_BC_set}
    for read in bam.fetch(insertion_dict['chrom'], insertion_dict['start'], insertion_dict['end']):
        cell = read.get_tag('BC')
        if cell not in cell_BC_set:
            continue
        if not correct_orientation(read, insertion_dict['strand']):
            continue
        seq = read.get_tag('LS')
        if seq in seq_to_count_dict:
            seq_info = seq_to_count_dict[seq]
            tagmentation_site = read.reference_end if read.is_reverse else read.reference_start
            if tagmentation_site in A_count_dict[cell][seq_info[0]]:
                A_count_dict[cell][seq_info[0]][tagmentation_site].append(seq_info[1])
            else:
                A_count_dict[cell][seq_info[0]][tagmentation_site] = [seq_info[1]]
        else:
            pre_string_list, A_count_list = summarize_sequence_and_count_A_tracts(seq, 5,  min_len, break_len)
            seq_summary = ','.join(pre_string_list)
            if seq_summary not in allele_summary_set and seq_summary not in seq_summaries_to_correct:
                continue
            tagmentation_site = read.reference_end if read.is_reverse else read.reference_start
            if tagmentation_site in A_count_dict[cell][seq_summary]:
                A_count_dict[cell][seq_summary][tagmentation_site].append(A_count_list)
            else:
                A_count_dict[cell][seq_summary][tagmentation_site] = [A_count_list]
            seq_to_count_dict[seq] = (seq_summary, A_count_list)

    A_counter_dict = {cell: {allele_summary_to_id[a]: {} for a in allele_list} for cell in cell_BC_set}
    
    A_count_len_dict = {cell: {} for cell in cell_BC_set}
    
    for cell, allele_dict in A_count_dict.items():
        for a, tag_site_dict in allele_dict.items():
            if a not in allele_summary_set:
                continue
            for p, A_count_list in tag_site_dict.items():
                if p in A_count_len_dict[cell]:
                    A_count_len_dict[cell][p][a] = len(A_count_list)
                else:
                    A_count_len_dict[cell][p] = {a: len(A_count_list)}
    
    A_count_allow_list = {cell: {} for cell in cell_BC_set}
    
    A_count_cell_max_allele = {cell: {} for cell in cell_BC_set}
    
    for cell, position_dict in A_count_len_dict.items():
        for p, a_dict in position_dict.items():
            a_dict_pruned = {}
            for a, v in a_dict.items():
                if a in allele_summary_set:
                    a_dict_pruned[a] = v
            if len(a_dict_pruned) == 0:
                continue
            if len(a_dict_pruned) == 1:
                A_count_cell_max_allele[cell][p] = allele_summary_to_id[list(a_dict_pruned.keys())[0]]
            max_allele = max(a_dict_pruned, key=a_dict_pruned.get)
            A_count_cell_max_allele[cell][p] = max_allele
            A_count_allow_list[cell][p] = [allele_summary_to_id[max_allele]]
            for a, a_count in a_dict_pruned.items():
                if a == max_allele:
                    continue
                cond_prob_val = allele_call[a]['conditional_probs'][max_allele]
                if cond_prob_val*a_dict_pruned[max_allele]*200 < a_count:
                    A_count_allow_list[cell][p].append(allele_summary_to_id[a])
    
    for cell, allele_dict in A_count_dict.items():
        for a, tag_site_dict in allele_dict.items():
            if a not in allele_summary_set:
                continue
            a_idx = allele_summary_to_id[a]
            for p, A_count_list in tag_site_dict.items():
                if len(A_count_list) < threshold:
                    continue
                if p in A_count_allow_list[cell]:
                    if a_idx not in A_count_allow_list[cell][p]:
                        continue
                A_counter_dict[cell][a_idx][p] = {}
                for i in range(a.count(',')):
                    A_counter_dict[cell][a_idx][p][i] = Counter([l[i] for l in A_count_list])
    
    for cell, allele_dict in A_count_dict.items():
        for a, tag_site_dict in allele_dict.items():
            if a not in seq_summaries_to_correct or a in allele_summary_set:
                continue
            for p, A_count_list in tag_site_dict.items():
                if p in A_count_cell_max_allele[cell]:
                    original_allele = A_count_cell_max_allele[cell][p]
                    a_idx = allele_summary_to_id[original_allele]
                    if a not in seq_to_correct_per_allele[original_allele]:
                        continue
                    if seq_to_correct_per_allele[original_allele][a] is None:
                        continue
                    if len(seq_to_correct_per_allele[original_allele][a].keys()) != len(A_count_list[0]):
                        continue
                    for i, c in seq_to_correct_per_allele[original_allele][a].items():
                        A_counter = Counter([l[i]+c for l in A_count_list])
                    if p not in A_counter_dict[cell][a_idx]:
                        continue
                    for j in range(original_allele.count(',')):
                        A_counter_dict[cell][a_idx][p][j].update(A_counter)
    
    return ins_key, A_counter_dict

def get_histogram_dfs(bamfile, cell_BC_set, insertion_dict, allele_list, seq_to_correct_per_allele, allele_call, ins_key, threshold, min_len, break_len):
    tup = get_A_counter_dict_w_corrections(bamfile, cell_BC_set, insertion_dict, allele_list, seq_to_correct_per_allele, allele_call, ins_key, threshold, min_len, break_len)
    ins_key, A_counter_dict = tup

    df_list = []

    for cell, alleles_count in A_counter_dict.items():
        for a, allele_count in alleles_count.items():
            for tag_site, A_dict in allele_count.items():
                for tract_i, A_counter in A_dict.items():
                    A_counter_series = pd.Series(A_counter)
                    A_counter_series.name = (cell, ins_key, a, tract_i, tag_site)
                    df_list.append(A_counter_series)
    if len(df_list) > 0:
        df = pd.concat(df_list, axis=1).T.reindex(pd.Index(list(range(3,50))), axis=1)
        df = df.fillna(0)
    else:
        df = pd.DataFrame(columns = pd.Index(list(range(3,50))))
        df = df.fillna(0)
    df['read_num'] = df.sum(axis=1)

    old_read_num = df['read_num']
    df = df.div(df['read_num'], axis=0)
    df['read_num'] = old_read_num

    df.index = pd.MultiIndex.from_tuples(df.index, names=['cell', 'ins_key', 'a_key', 't_key', 'tag_site'])
    try:
        test = add_kd_calling(df)
        test['hist_peak'] = test[range(3,50)].idxmax(axis=1)
    except ValueError:
        return ins_key, df
    
    return ins_key, test

from collections import Counter
def find_biallelic_boundaries(peak_df):
    biallelic_boundaries = {}
    num_peaks = peak_df['read_num']
    num_peaks_sorted = num_peaks.sort_values()
    num_peaks_sorted = num_peaks_sorted.reset_index()
    num_peaks_sorted.index = num_peaks_sorted['hist_peak']
    biallelic_peaks = None
    num_peaks_sorted = num_peaks_sorted.astype(int)
    if num_peaks_sorted.shape[0] > 1:
        num_peaks_sorted_subset = num_peaks_sorted.iloc[-2:]
        peaks_total = num_peaks_sorted_subset.sum()['read_num']
        if num_peaks_sorted_subset.max()['read_num']/peaks_total < 0.6:
            biallelic_peaks = num_peaks_sorted_subset.index.values
        else:
            return None, None, None, None
        biallelic_peaks = np.sort(biallelic_peaks)
        min_peak = min(biallelic_peaks)
        difference = int(np.ceil((biallelic_peaks[1]-biallelic_peaks[0]-1)/2))
        total_diff = biallelic_peaks[1]-biallelic_peaks[0]
        lower_top = biallelic_peaks[0]+difference
        bottom_lower = biallelic_peaks[1]-difference
        bottom_a = np.arange(lower_top+1)
        top_a = np.arange(bottom_lower,50)
        for p in num_peaks_sorted.index.values:
            if p in bottom_a:
                biallelic_boundaries[p] = 0
            else:
                biallelic_boundaries[p] = 1
    else:
        return None, None, None, None
    return biallelic_peaks, biallelic_boundaries, min_peak, total_diff

def make_combined_boundaries_dict(boundaries_dict, test_peaks):
    test_peaks_reset = test_peaks.reset_index()
    t_key_to_order = {t_key:i for i,t_key in enumerate(boundaries_dict.keys())}
    assignment_combinations = Counter()
    for cell, cell_df in test_peaks_reset.groupby('cell'):
        for tag_site, tag_site_df in cell_df.groupby('tag_site'):
            assignment_dict = {}
            l = [0 for i in range(len(t_key_to_order))]
            for idx, row in tag_site_df.iterrows():
                if row['t_key'] in boundaries_dict.keys():
                    assignment_dict[row['t_key']] = boundaries_dict[row['t_key']]['biallelic_boundaries'][row['hist_peak']]
            if len(assignment_dict) != len(t_key_to_order):
                continue
            for t_key, assignment in assignment_dict.items():
                l[t_key_to_order[t_key]] = assignment
            assignment_combinations.update({tuple(l):1})
    return assignment_combinations, t_key_to_order

def scan_for_biallelic_signal(test_peaks):
    boundaries_dict = {}
    for t_key, df in test_peaks.groupby('t_key'):
        peak_df = df.groupby('hist_peak').count()
        biallelic_peaks, biallelic_boundaries, min_peak, difference = find_biallelic_boundaries(peak_df)
        if biallelic_boundaries is None:
            continue
        else:
            boundaries_dict[t_key] = {'biallelic_peaks':biallelic_peaks, 'biallelic_boundaries':biallelic_boundaries, 'min_peak':min_peak, 'difference':difference}
    if len(boundaries_dict) > 1:
        assignment_combinations, t_key_to_order = make_combined_boundaries_dict(boundaries_dict, test_peaks)
    else:
        assignment_combinations, t_key_to_order = None, None
    return boundaries_dict, assignment_combinations, t_key_to_order


def get_allele_index(a_key, tract_site, tract_categorization_cell):
    if len(tract_categorization_cell) == 0:
        return a_key
    else:
        if tract_site in tract_categorization_cell:
            return tract_categorization_cell[tract_site]
        else:
            return None

def get_final_stats_to_write(test, ins_key, insertion_dict):
    n_alleles = test.groupby('a_key').count().shape[0]
    if n_alleles == 1:
        t_key, biallelic_boundaries,insertion_difference = scan_for_biallelic_signal(test)
        if t_key is None:
            return None, None
        print(t_key, biallelic_boundaries,insertion_difference)
        if insertion_difference < 3:
            return None, None
    else:
        biallelic_boundaries = None
    test_flat = test.reset_index()
    tract_categorization = {cell: {} for cell in test_flat['cell'].unique()}
    
    if biallelic_boundaries is not None:
        print(t_key, biallelic_boundaries)
        for idx, row in test_flat.iterrows():
            if row['t_key'] == t_key:
                tract_categorization[row['cell']][row['tag_site']] = biallelic_boundaries[int(row['hist_peak'])]
        n_alleles = 2
    if n_alleles == 1 and insertion_dict['type'] == 'KR':
        return None, None
    if biallelic_boundaries is None:
        max_peak = test.groupby('a_key').apply(lambda df: df.groupby('t_key').apply(lambda df2: df2.groupby('kd_allele').apply(lambda df3: df3.groupby('hist_peak').count().idxmax())))['kd_allele']
        try:
            original_alleles = test.groupby('a_key').apply(lambda df: df.groupby('t_key').apply(lambda df2: df2.groupby('kd_allele').count().idxmax()))['hist_peak']
        except ValueError:
            return None, None

        hist_peak_dict = {}
        for k, kd_allele in original_alleles.items():
            k_tup = (k[0],k[1], kd_allele)
            hist_peak_dict[k] = max_peak[k_tup]
    
    if biallelic_boundaries is not None:
        tract_stats = {cell: {} for cell in test_flat['cell'].unique()}
        for idx, row in test_flat.iterrows():
            allele_idx = get_allele_index(row['a_key'], row['tag_site'], tract_categorization[row['cell']])
            if allele_idx is None:
                continue

            if allele_idx not in tract_stats[row['cell']]:
                tract_stats[row['cell']][allele_idx] = {}
                tract_stats[row['cell']][allele_idx][row['t_key']] = {}
                tract_stats[row['cell']][allele_idx][row['t_key']][row['tag_site']] = (row['kd_allele'], row['hist_peak'], row['read_num'])
            else:
                if row['t_key'] not in tract_stats[row['cell']][allele_idx]:
                    tract_stats[row['cell']][allele_idx][row['t_key']] = {}
                    tract_stats[row['cell']][allele_idx][row['t_key']][row['tag_site']] = (row['kd_allele'], row['hist_peak'], row['read_num'])
                else:
                    tract_stats[row['cell']][allele_idx][row['t_key']][row['tag_site']] = (row['kd_allele'], row['hist_peak'], row['read_num'])

        original_alleles, hist_peak_dict = find_original_alleles(tract_stats)

    tract_stats = {cell: {} for cell in test_flat['cell'].unique()}
    for idx, row in test_flat.iterrows():
        allele_idx = get_allele_index(row['a_key'], row['tag_site'], tract_categorization[row['cell']])
        if allele_idx is None:
            continue
        if row['hist_peak'] == hist_peak_dict[(allele_idx, row['t_key'])]:
            kd_allele = original_alleles[(allele_idx, row['t_key'])]
        else:
            kd_allele = row['kd_allele']
        if allele_idx not in tract_stats[row['cell']]:
            tract_stats[row['cell']][allele_idx] = {}
            tract_stats[row['cell']][allele_idx][row['t_key']] = {}
            tract_stats[row['cell']][allele_idx][row['t_key']][row['tag_site']] = (kd_allele, row['hist_peak'], row['read_num'])
        else:
            if row['t_key'] not in tract_stats[row['cell']][allele_idx]:
                tract_stats[row['cell']][allele_idx][row['t_key']] = {}
                tract_stats[row['cell']][allele_idx][row['t_key']][row['tag_site']] = (kd_allele, row['hist_peak'], row['read_num'])
            else:
                tract_stats[row['cell']][allele_idx][row['t_key']][row['tag_site']] = (kd_allele, row['hist_peak'], row['read_num'])
    
    conservative_fun = make_determine_function(original_alleles)
    final_stats = find_most_likely_length(tract_stats, conservative_fun)

    cell_detection_stats = {i: set() for i in range(n_alleles)}
    
    final_stats_unique_key = {}
    for cell, cell_stats in final_stats.items():
        final_stats_unique_key[cell] = {}
        for a_key, a_stats in cell_stats.items():
            if a_key in cell_detection_stats:
                cell_detection_stats[a_key].add(cell)
            for t_key, v in a_stats.items():
                final_stats_unique_key[cell]['{}_{}_{}'.format(ins_key, a_key,t_key)] = v
    return final_stats_unique_key, cell_detection_stats

def find_original_alleles(tract_stats):
    kd_allele_per_tract = {}
    kd_allele_and_hist_peak = {}
    for cell, cell_stats in tract_stats.items():
        for allele_idx, allele_stats in cell_stats.items():
            if allele_idx not in kd_allele_per_tract:
                kd_allele_per_tract[allele_idx] = {}
                kd_allele_and_hist_peak[allele_idx] = {}
            for tract_idx, tag_stats in allele_stats.items():
                if tract_idx not in kd_allele_per_tract[allele_idx]:
                    kd_allele_per_tract[allele_idx][tract_idx] = Counter()
                    kd_allele_and_hist_peak[allele_idx][tract_idx] = {}
                for tag_site, tag_tup in tag_stats.items():
                    if tag_tup[0] not in kd_allele_and_hist_peak[allele_idx][tract_idx]:
                        kd_allele_and_hist_peak[allele_idx][tract_idx][tag_tup[0]] = Counter({tag_tup[1]:1})
                    else:
                        kd_allele_and_hist_peak[allele_idx][tract_idx][tag_tup[0]].update(Counter({tag_tup[1]:1}))
                    kd_allele_per_tract[allele_idx][tract_idx].update({tag_tup[0]:1})
    original_alleles = {}
    hist_peak_dict = {}
    for allele_idx, allele_stats in kd_allele_per_tract.items():
        for tract_idx, kd_per_tract in allele_stats.items():
            most_common_kd = kd_per_tract.most_common()[0]
            original_alleles[(allele_idx, tract_idx)] = most_common_kd[0]
            hist_peaks = kd_allele_and_hist_peak[allele_idx][tract_idx][most_common_kd[0]].most_common()[0]
            hist_peak_dict[(allele_idx, tract_idx)] = hist_peaks[0]
    return original_alleles, hist_peak_dict

def max_reads_fun(tract_d):
    c = Counter()
    for tag_site, tag_tup in tract_d.items():
        c.update({tag_tup[0]: tag_tup[2]})
    return c.most_common()[0][0]

def make_determine_function(original_lengths):
    def return_value(tract_d, allele_idx, tract_idx):
        c = Counter()
        for tag_site, tag_tup in tract_d.items():
            if tag_tup[0] == original_lengths[(allele_idx,tract_idx)]:
                return tag_tup[0]
            c.update({tag_tup[0]: tag_tup[2]})
        return c.most_common()[0][0]
    return return_value

def find_most_likely_length(tract_stats, determine_fun):
    final_stats = {}
    for cell, stats_dict in tract_stats.items():
        final_stats[cell] = {}
        for allele_idx, allele_d in stats_dict.items():
            final_stats[cell][allele_idx] = {}
            for tract_idx, tract_d in allele_d.items():
                final_stats[cell][allele_idx][tract_idx] = determine_fun(tract_d, allele_idx, tract_idx)
    return final_stats

def add_root_to_df(df):
    root = df.median()
    root.name = 'ROOT'
    df = df.append(root)
    return df 

import seaborn as sns
def get_median_of_tracts(df):
    s = df.groupby('a_key').apply(lambda df2: df2.groupby('t_key').apply(lambda df3: df3.groupby('kd_allele').count().idxmax()))['hist_peak']
    s.name = 'zygote_tract'
    return s

def get_false_positive_rate(key, histogram_df):
    n_alleles = histogram_df.groupby('a_key').count().shape[0]
    if n_alleles == 1:
        return key, None, None, None
    try:
        zygote_tracts = get_median_of_tracts(histogram_df)
    except (KeyError, ValueError):
        return key, None, None, None
    total_obs = {t:0 for t in zygote_tracts.to_dict().keys()}
    agreeing_obs = {t:0 for t in zygote_tracts.to_dict().keys()}
    for cell, df_cell in histogram_df.groupby('cell'):
        for allele, df_allele in df_cell.groupby('a_key'):
            for tract, df_tract in df_allele.groupby('t_key'):
                tract_lengths = df_tract['kd_allele'].dropna()
                if len(tract_lengths) < 2:
                    continue
                is_zygote_tract = tract_lengths.isin([zygote_tracts[allele,tract]])
                total_obs[allele,tract] += is_zygote_tract.shape[0]
                agreeing_obs[allele,tract] += is_zygote_tract.sum()
    return key, zygote_tracts, total_obs, agreeing_obs

def get_mutated_cells_fun(zygote_tracts):
    def get_mutated_cells(df):
        cell_list = []
        for cell, cell_df in df.groupby('cell'):
            for a_key, a_df in cell_df.groupby('a_key'):
                for t_key, a_df in a_df.groupby('t_key'):
                    if all([kd_allele != zygote_tracts[a_key,t_key] for kd_allele in a_df['kd_allele'].dropna()]) and len(a_df['kd_allele'].dropna()) > 1 and zygote_tracts[a_key,t_key] < 14 :
                        if len(set(a_df['kd_allele'])) == 1:
                            cell_list.append(cell)
        cell_list = list(set(cell_list))
        return cell_list
    return get_mutated_cells

def get_cells_with_match(df, a_key, t_key, kd_allele):
    df_reset = df.reset_index()
    df_subset = df_reset[(df_reset['a_key'] == a_key)&(df_reset['t_key'] == t_key)&(df_reset['kd_allele'] == kd_allele)]
    cell_list = []
    for cell in list(set(df_subset['cell'])):
        df_cell = df_reset[(df_reset['a_key'] == a_key)&(df_reset['t_key'] == t_key)&(df_reset['cell'] == cell)]
        if len(set(df_cell['kd_allele'])) == 1:
            cell_list.append(cell)
    return cell_list

def get_cells_w_value(df, a_key,t_key):
    df_reset = df.reset_index()
    df_subset = df_reset[(df_reset['a_key'] == a_key)&(df_reset['t_key'] == t_key)&(df_reset['kd_allele'].notna())]
    return df_subset['cell'].unique()

def get_mutated_cells_fun(zygote_tracts):
    def get_mutated_cells(df):
        tract_mutation_dict = {}
        for cell, cell_df in df.groupby('cell'):
            for a_key, a_df in cell_df.groupby('a_key'):
                for t_key, a_df in a_df.groupby('t_key'):
                    if any([kd_allele != zygote_tracts[a_key,t_key] for kd_allele in a_df['kd_allele'].dropna()]) and len(a_df['kd_allele'].dropna()) > 0:
                        kd_allele_list = list(set(a_df['kd_allele']))
                        if len(kd_allele_list) == 1:
                            if (a_key,t_key, kd_allele_list[0]) in tract_mutation_dict:
                                tract_mutation_dict[(a_key,t_key, kd_allele_list[0])].append(cell)
                            else:
                                tract_mutation_dict[(a_key,t_key, kd_allele_list[0])] = [cell]
        tract_mutation_dict = {k: list(set(l)) for k,l in tract_mutation_dict.items()}
        return tract_mutation_dict
    return get_mutated_cells
import sys
sys.path.append("sequencing/calling/")
from sklearn.neighbors import KDTree

import pickle
with open('hist_map.pickle', 'rb') as f:
    hist_map = pickle.load(f)
def cosine_score_to_dist(alpha):
    return 2*np.sin(np.arccos(1-alpha)/2)

def dist_to_cosing_score(d):
    return 1-np.cos(2*np.arcsin(d/2))
from multi_hists import MonoSimulatedHistogram
from sklearn.neighbors import KDTree
LEAF_SIZE = 100 # optimization shows 50-110 are almost the same
L2normalized_vecs = [(sh._vec/np.linalg.norm(sh._vec, ord=2))[3:50] for sh in hist_map]
kdtree = KDTree(np.array(L2normalized_vecs), leaf_size=LEAF_SIZE)
from collections import namedtuple

HistKey = namedtuple('HistKey',['cell', 'ins_key', 'a_key', 't_key', 'tag_site'])
def extract_normalized_vectors(df3_40):
    vech_keys = []
    vech_values = []
    for k, r in df3_40.iterrows():
        vech = np.array(r)
        vech_normed = vech/np.linalg.norm(vech, ord=2)
        if np.any(np.isnan(vech_normed)):
            print('warning nans in {}'.format(k))
            continue
        vech_keys.append(HistKey(*k))
        vech_values.append(vech_normed)
    return vech_keys, vech_values

Call = namedtuple('Call',['hist','conf'])
def map_tree_query(ind, dist, vech_keys):
    called = {}
    for (ind0, dist0, k) in zip(ind, dist, vech_keys):
        if len(ind0)!=0:
            imin = np.argmin(dist0)
            called[k] = Call(hist_map[ind0[imin]], dist_to_cosing_score(dist0[imin]))
    return called
def add_kd_calling(hists_df, conf=0.05): # conf = 1-correlation I reccoment a correlation of at least 0.95
    vech_keys, vech_values = extract_normalized_vectors(hists_df[list(range(3,50))])
    print("starting tree query")
    CONF_RADIOS_THRESHOLD = cosine_score_to_dist(conf)
    (ind, dist) = kdtree.query_radius(vech_values, r=CONF_RADIOS_THRESHOLD, return_distance=True)
    print("finished tree query")
    called = map_tree_query(ind, dist, vech_keys)
    flat_called = dict()
    for k, call_item in called.items():
        flat_called[k.cell, k.ins_key, k.a_key, k.t_key, k.tag_site] = [call_item.hist._ms_len, call_item.conf]
    kd_calling_df = pd.DataFrame.from_dict(flat_called).T
    kd_calling_df.index.set_names(['cell', 'ins_key', 'a_key', 't_key', 'tag_site'], inplace=True)
    kd_calling_df.columns=['kd_allele', 'kd_conf']
    return hists_df.join(kd_calling_df, rsuffix='_kd')
def get_best_repeat_number(df):
    if df.shape[0] == 1:
        return df['kd_allele']
    else:
        return df.groupby('kd_allele').apply(lambda df2: df)


## Tries to determine the minimum number of operations to obtain the original allele, and how to change the corresponding tract lengths.
def calculate_minimum_operations(source_motif, source_flank, destination_motif, destination_flank):
    destination_operation_list = get_operation_list(destination_motif)
    destination_motif_list = destination_motif.split(',')
    source_motif_list = source_motif.split(',')
    n = len(destination_motif_list)-1
    
    i = 0
    #print(n)
    action_list = []
    j = 0
    while i <= n:
        #print(source_motif_list)
        #print(i, n)
        ## Tries to figure out what the next move should be.
        action = determine_next_action(source_motif_list, destination_motif_list, source_flank, destination_flank,destination_operation_list , i, n)
        #print(action)
        
        if (action is not None):
            action_list.append(action)
            ## Some actions lead to non-identifiable tract lengths, and we should just give up.
            if action[0] in ['abort', 'truncate'] or action[2] == 'abort':
                return action_list
            ## Figure out of the edit action affect the tract length.
            source_motif_list = apply_action_list(source_motif_list, [action])
            if j == 10:
                action_list.append(('infinite',0,0,0))
                return action_list
            j += 1
        else:
            j = 0
            i += 1    
    return action_list
def truncate_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_motif_list = source_motif_list[:i]
    return source_motif_list
def apply_action_list(source_motif_list, action_list):
    for action_tuple in action_list:
        source_motif_list = allowed_actions[action_tuple[0]](source_motif_list, action_tuple)
    return source_motif_list
def edit_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_seq = source_motif_list[i]
    source_seq = source_seq[:j] + seq + source_seq[j+1:]
    source_motif_list[i] = source_seq
    return source_motif_list
def prepend_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_seq = source_motif_list[i]
    source_seq = seq + source_seq
    if j == 0:
        source_motif_list[i] = source_seq
    else:
        source_motif_list[i] = source_seq[:j]
    return source_motif_list
def append_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_seq = source_motif_list[i]
    source_seq = source_seq + seq
    source_motif_list[i] = source_seq
    return source_motif_list
def split_action(source_motif_list, action_tuple):
    action, i, j, k = action_tuple
    source_seq = source_motif_list[i]
    source_seq_1 = seq[:j]
    source_seq_2 = seq[k:]
    source_motif_list[i] = source_seq_1
    source_motif_list.insert(i+1, source_seq_2)
    return source_motif_list
def insert_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_motif_list.insert(i, seq)
    return source_motif_list
def insert_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_motif_list.insert(i, seq)
    return source_motif_list
def remove_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_motif_list = source_motif_list[:i] + source_motif_list[(i+1):]
    return source_motif_list
def strip_action(source_motif_list, action_tuple):
    action, i, j, seq = action_tuple
    source_seq = source_motif_list[i]
    if seq == 'r':
        source_seq = source_seq[:j]
    else:
        source_seq = source_seq[j:]
    source_motif_list[i] = source_seq
    return source_motif_list

allowed_actions = {'edit': edit_action, 'prepend': prepend_action, 'append': append_action, 'split': split_action, 'insert': insert_action, 'remove': remove_action, 'strip': strip_action, 'truncate': truncate_action}
def get_op_seq(seq):
    op_seq = ''
    for j, c in enumerate(seq):
        if j == 0:
            op_seq += c
        elif c == 'A':
            op_seq += c
        else:
            break
    return op_seq
def get_end_consensus(seqs):
    end_flank_list = []
    for seq in seqs:
        flanking_sequences, A_count_list = split_sequence(seq)
        end_flank_list.append(flanking_sequences[-1])
    cons_seq = get_consensus_sequence(end_flank_list)
    return cons_seq
def get_tag_site_dist(bamfile, cell_BC_set, insertion_dict, key, min_len, break_len):
    tagmentation_site_per_allele_counter = collect_tag_site_info(bamfile, cell_BC_set, insertion_dict, min_len, break_len)
    tag_site_dists = Counter()
    for cell, d in tagmentation_site_per_allele_counter.items():
        tag_site_dists.update({len(d):1})
    return tag_site_dists

def allele_summary_distance(summary1, summary2):
    # Allele summaries are comma-separated segments around polyA tracts (N segments = N-1 tracts).
    # Needleman-Wunsch over segments: match cost = Levenshtein, indel of segment s costs len(s) + 1
    # (one polyA tract structural change + len(s) flanking characters). Returns (n_tract_diff, seq_edit_dist).
    parts1 = summary1.split(',')
    parts2 = summary2.split(',')
    m, n = len(parts1), len(parts2)
    total = lambda s: s[0] + s[1]
    dp = [[None] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = (0, 0)
    for i in range(1, m + 1):
        prev = dp[i - 1][0]
        dp[i][0] = (prev[0] + 1, prev[1] + len(parts1[i - 1]))
    for j in range(1, n + 1):
        prev = dp[0][j - 1]
        dp[0][j] = (prev[0] + 1, prev[1] + len(parts2[j - 1]))
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            lev = Levenshtein.distance(parts1[i - 1], parts2[j - 1])
            match_state = (dp[i - 1][j - 1][0], dp[i - 1][j - 1][1] + lev)
            delete_state = (dp[i - 1][j][0] + 1, dp[i - 1][j][1] + len(parts1[i - 1]))
            insert_state = (dp[i][j - 1][0] + 1, dp[i][j - 1][1] + len(parts2[j - 1]))
            dp[i][j] = min([match_state, delete_state, insert_state], key=total)
    return dp[m][n]

def build_insertion_summary(insertion_information, res_dict, allele_calls,
                             histogram_dfs_res_dict, peak_based_calling_result_dict,
                             het_sites_dict, snp_cpm_dict,
                             phase_allele_snv_dict, phase_biallele_snv_dict):
    rows = []
    for ins_key, ins_info in insertion_information.items():
        row = {
            'ins_key': ins_key,
            'chrom': ins_info.get('chrom'),
            'start': ins_info.get('start'),
            'end': ins_info.get('end'),
            'strand': ins_info.get('strand'),
            'type': ins_info.get('type'),
        }

        # Allele detection stats
        if ins_key in res_dict:
            rd = res_dict[ins_key]
            row['cell_w_counts'] = rd['cell_w_counts']
            row['cell_w_tag_site'] = rd['cell_w_tag_site']
            row['n_allele_candidates'] = len(rd['allele_counter'])
        else:
            row['cell_w_counts'] = None
            row['cell_w_tag_site'] = None
            row['n_allele_candidates'] = None

        # Sequence-based allele calls
        if ins_key in allele_calls:
            ac = allele_calls[ins_key]
            row['passed_cell_threshold'] = True
            row['n_alleles_sequence'] = len(ac)
            row['allele_summaries'] = ';'.join(ac.keys())
            row['allele_fractions'] = ';'.join(
                '{:.3f}'.format(v['fraction_tags']) for v in ac.values()
            )
            allele_keys = list(ac.keys())
            if len(allele_keys) > 1:
                dist_parts = []
                for i, j in combinations(range(len(allele_keys)), 2):
                    n_tract_diff, seq_dist = allele_summary_distance(allele_keys[i], allele_keys[j])
                    if n_tract_diff == 0:
                        dist_parts.append('{}-{}:seq={}'.format(i, j, seq_dist))
                    else:
                        dist_parts.append('{}-{}:tracts={},seq={}'.format(i, j, n_tract_diff, seq_dist))
                row['allele_edit_distances'] = ';'.join(dist_parts)
            else:
                row['allele_edit_distances'] = None
        else:
            row['passed_cell_threshold'] = False
            row['n_alleles_sequence'] = None
            row['allele_summaries'] = None
            row['allele_fractions'] = None
            row['allele_edit_distances'] = None

        # Histogram / kd calling
        if ins_key in histogram_dfs_res_dict:
            hdf = histogram_dfs_res_dict[ins_key]
            row['n_cells_histogram'] = (
                hdf.index.get_level_values('cell').nunique() if len(hdf) > 0 else 0
            )
        else:
            row['n_cells_histogram'] = None

        # Peak-based biallelic signal
        if ins_key in peak_based_calling_result_dict:
            bd, _, _ = peak_based_calling_result_dict[ins_key]
            row['biallelic_signal_detected'] = True
            row['biallelic_t_keys'] = ';'.join(str(t) for t in bd.keys())
            row['biallelic_peaks'] = ';'.join(
                '{}:{},{}'.format(t, v['biallelic_peaks'][0], v['biallelic_peaks'][1])
                for t, v in bd.items()
            )
        elif ins_key in allele_calls and len(allele_calls[ins_key]) == 1:
            # Single allele was processed but no biallelic signal found
            row['biallelic_signal_detected'] = False
            row['biallelic_t_keys'] = None
            row['biallelic_peaks'] = None
        else:
            # Not applicable: multiple sequence alleles already detected, or cell threshold not met
            row['biallelic_signal_detected'] = None
            row['biallelic_t_keys'] = None
            row['biallelic_peaks'] = None

        # Heterozygous SNP sites
        row['n_het_sites'] = len(het_sites_dict[ins_key]) if ins_key in het_sites_dict else None

        # SNP CPM calling
        if ins_key in snp_cpm_dict:
            snp_res = snp_cpm_dict[ins_key]
            if snp_res[0] is None:
                row['snp_cpm_n_phased_variants'] = 0
                row['snp_cpm_unequal_len'] = None
            else:
                variants_phased, unequal_len = snp_res[0], snp_res[1]
                row['snp_cpm_n_phased_variants'] = sum(len(v) for v in variants_phased.values())
                row['snp_cpm_unequal_len'] = unequal_len
        else:
            row['snp_cpm_n_phased_variants'] = None
            row['snp_cpm_unequal_len'] = None

        # Phase: sequence alleles vs SNVs
        if ins_key in phase_allele_snv_dict:
            phasv = phase_allele_snv_dict[ins_key]
            p_vals = [v[2] for v in phasv.values() if v[2] is not None]
            row['phase_allele_snv_min_p'] = min(p_vals) if p_vals else None
            row['phase_allele_snv_sites'] = ';'.join(
                '{}(chi2={:.2f},p={:.2e})'.format(pos, v[1], v[2])
                for pos, v in phasv.items() if v[1] is not None
            )
        else:
            row['phase_allele_snv_min_p'] = None
            row['phase_allele_snv_sites'] = None

        # Phase: peak-based biallelic signal vs SNVs
        if ins_key in phase_biallele_snv_dict:
            pbsv = phase_biallele_snv_dict[ins_key]
            p_vals = [v[2] for v in pbsv.values() if v[2] is not None]
            row['phase_biallele_snv_min_p'] = min(p_vals) if p_vals else None
            row['phase_biallele_snv_sites'] = ';'.join(
                '{}(chi2={:.2f},p={:.2e})'.format(pos, v[1], v[2])
                for pos, v in pbsv.items() if v[1] is not None
            )
        else:
            row['phase_biallele_snv_min_p'] = None
            row['phase_biallele_snv_sites'] = None

        rows.append(row)

    if not rows:
        return pd.DataFrame().rename_axis('ins_key')
    df = pd.DataFrame(rows).set_index('ins_key')
    return df


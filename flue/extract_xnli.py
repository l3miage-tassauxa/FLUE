"""
Hang Le (hangtp.le@gmail.com)
"""
import os
import numpy as np
import math
import random
import re
import csv
import argparse

import sys
sys.path.append(os.getcwd())

from tools.clean_text import cleaner
from xlm.utils import bool_flag


def get_labels(line, do_lower=False):
    """
    Input: line
    Returns pairs of sentences and corresponding labels
    """
    sent1, sent2, label = line.split('\t')
    sent1 = cleaner(sent1, rm_new_lines=True, do_lower=do_lower)
    sent2 = cleaner(sent2, rm_new_lines=True, do_lower=do_lower)
    # For Hugging Face, combine sentences with [SEP] token
    combined_text = sent1 + " [SEP] " + sent2
    label = label.strip()

    return combined_text, label


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--indir', type=str, help='Path to processed data directory')
    parser.add_argument('--outdir', type=str, default=None, help='Path to output data directory')
    parser.add_argument('--do_lower', type=bool_flag, default='False', help='True if do lower case, False otherwise.')

    args = parser.parse_args()

    input_path = os.path.expanduser(args.indir)
    output_path = os.path.expanduser(args.outdir) if args.outdir else input_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    splts = ['valid', 'test', 'train']
    lang = 'fr'

    for s in splts:
        sent_pairs = []
        labels = []

        with open(os.path.join(input_path, lang+'.raw.'+s), 'rt', encoding='utf-8') as f_in:
            next(f_in)
            # Create TSV file for Hugging Face conversion
            with open(os.path.join(output_path, '{}.tsv'.format(s)), 'w') as f_out:
                tsv_output = csv.writer(f_out, delimiter='\t')
                for line in f_in:
                    combined_text, label = get_labels(line, do_lower=args.do_lower)
                    sent_pairs.append(combined_text)
                    labels.append(label)

                    tsv_output.writerow([combined_text, label])

        assert len(sent_pairs) == len(labels)

        print('Finished writing {}.tsv to {}. Neutral/Contradiction/Entailment: {}/{}/{}'.format(s, 
                                                                                                output_path, 
                                                                                                labels.count('neutral'),
                                                                                                labels.count('contradiction'),
                                                                                                labels.count('entailment')))

if __name__ == "__main__":
    main()
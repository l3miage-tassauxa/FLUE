# binarize_txt_to_pth.py
import sys
import torch

infile = sys.argv[1]
outfile = sys.argv[2]

with open(infile, 'r', encoding='utf-8') as f:
    lines = [line.rstrip('\n') for line in f]

torch.save(lines, outfile)
#!/usr/bin/env python3
"""
Convert TSV files to CSV format for Hugging Face compatibility.
This script properly handles text with commas, quotes, and newlines.
"""

import csv
import sys
import os
from pathlib import Path

def convert_tsv_to_csv(tsv_file, csv_file):
    """
    Convert a TSV file to CSV format with proper escaping.
    """
    print(f"Converting {tsv_file} to {csv_file}")
    
    with open(tsv_file, 'r', encoding='utf-8') as infile, \
         open(csv_file, 'w', encoding='utf-8', newline='') as outfile:
        
        # Create readers and writers
        tsv_reader = csv.reader(infile, delimiter='\t')
        csv_writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        # Track if we need to add a header
        first_row = True
        has_header = False
        
        # Copy all rows
        for row in tsv_reader:
            if first_row:
                # Check if first row looks like a header (contains "Text" and "Label")
                if len(row) >= 2 and row[0].strip().lower() in ['text', 'Text'] and row[1].strip().lower() in ['label', 'Label']:
                    has_header = True
                    # Use lowercase headers for Hugging Face compatibility
                    csv_writer.writerow(['text', 'label'])
                else:
                    # Add header if missing
                    csv_writer.writerow(['text', 'label'])
                    csv_writer.writerow(row)
                first_row = False
            else:
                csv_writer.writerow(row)
    
    print(f"Successfully converted {tsv_file} to {csv_file}")
    if not has_header:
        print(f"  -> Added missing header to {csv_file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_tsv_to_csv.py <data_directory>")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    # Files to convert
    files_to_convert = ['train.tsv', 'valid.tsv', 'test.tsv']
    
    for filename in files_to_convert:
        tsv_path = data_dir / filename
        csv_path = data_dir / filename.replace('.tsv', '.csv')
        
        if tsv_path.exists():
            convert_tsv_to_csv(tsv_path, csv_path)
        else:
            print(f"Warning: {tsv_path} not found, skipping...")

if __name__ == "__main__":
    main()

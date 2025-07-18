#!/usr/bin/env python3
"""
Compute accuracy from Hugging Face prediction results and labels files.
"""

import argparse
import math

parser = argparse.ArgumentParser(description="Compute accuracy from Hugging Face predictions and labels files.")
parser.add_argument("--predictions_file", type=str, required=True, help="Path to the predictions file (e.g. predict_results_None.txt)")
parser.add_argument("--labels_file", type=str, required=True, help="Path to the gold labels file (e.g. test.label)")
args = parser.parse_args()

# Read predictions from Hugging Face output
preds = []
# XNLI label mapping
label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

with open(args.predictions_file) as f:
    next(f)  # Skip header line
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            pred_str = parts[1]  # Second column is the prediction
            # Handle both string and numeric predictions
            if pred_str.isdigit():
                pred = int(pred_str)
            elif pred_str in label_map:
                pred = label_map[pred_str]
            else:
                print(f"Warning: Unknown prediction label '{pred_str}', skipping...")
                continue
            preds.append(pred)

# Read gold labels
gold = []
with open(args.labels_file) as f:
    for line in f:
        label = line.strip()
        if label:
            # Handle both string and numeric labels
            if label.isdigit():
                gold.append(int(label))
            elif label in label_map:
                gold.append(label_map[label])
            else:
                print(f"Warning: Unknown gold label '{label}', skipping...")
                continue

# Calculate accuracy
total = min(len(preds), len(gold))
if total == 0:
    print("No predictions or labels found!")
    exit(1)

correct = sum([preds[i] == gold[i] for i in range(total)])
accuracy = correct / total

# Calculate 95% confidence interval
marge = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total)
print(f"Accuracy: {accuracy * 100:.2f}% ± {marge * 100:.2f}% on {total} examples (IC 95%).")

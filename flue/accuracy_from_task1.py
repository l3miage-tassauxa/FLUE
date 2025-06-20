import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Compute accuracy from logits and labels files.")
parser.add_argument("--logits_file", type=str, required=True, help="Path to the logits file (e.g. test.pred.14)")
parser.add_argument("--labels_file", type=str, required=True, help="Path to the gold labels file (e.g. test.label)")
args = parser.parse_args()

# Read logits and get predicted class (argmax)
preds = []
with open(args.logits_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        logits = [float(x) for x in parts[1].split(',')]
        pred = int(np.argmax(logits))
        preds.append(pred)

# Read gold labels
gold = []
with open(args.labels_file) as f:
    for line in f:
        label = line.strip()
        if label:
            gold.append(int(label))

# Compute accuracy
total = min(len(preds), len(gold))
correct = sum([preds[i] == gold[i] for i in range(total)])
accuracy = correct / total if total > 0 else 0.0

print(f"Accuracy: {accuracy * 100:.2f}% on {total} examples.")
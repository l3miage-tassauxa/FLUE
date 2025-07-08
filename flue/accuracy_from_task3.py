import numpy as np
from collections import Counter
import math

# File paths
logits_file = "./experiments/xnli_xlm_base_cased/dropout_0.1_lre_0.000005_lrp_0.000005/test.pred.9"
labels_file = "./flue/data/xnli/processed/test.label"

# Label order for classification
label_order = ['contradiction', 'neutral', 'entailment']
label_map = {label: idx for idx, label in enumerate(label_order)}
reverse_map = {v: k for k, v in label_map.items()}

# Logits from file test.pred.9
logit_lines = []
with open(logits_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        logits = [float(x) for x in parts[1].split(',')]
        logit_lines.append(logits)

# Labels from file test.label
gold_labels = []
with open(labels_file) as f:
    for line in f:
        label = line.strip()
        if label:
            gold_labels.append(label)

# Evaluate
correct = 0
total = min(len(logit_lines), len(gold_labels))
for i in range(total):
    gold = label_map[gold_labels[i]]
    pred = int(np.argmax(logit_lines[i]))
    if pred == gold:
        correct += 1

accuracy = correct / total
marge = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total) if total > 0 else 0.0
print(f"Accuracy: {accuracy * 100:.2f}% ± {marge * 100:.2f}% on {total} examples (IC 95%).")



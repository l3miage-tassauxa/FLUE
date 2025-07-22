#!/usr/bin/env python3
"""
Unified accuracy calculator for FLUE evaluation tasks.
Supports different input formats: XLM logits, Hugging Face predictions, and direct labels.
"""

import argparse
import math
import numpy as np
import os


def calculate_accuracy_with_confidence(predictions, gold_labels):
    """Calculate accuracy with 95% confidence interval."""
    total = min(len(predictions), len(gold_labels))
    if total == 0:
        print("No predictions or labels found!")
        return 0.0, 0.0, 0
    
    correct = sum([predictions[i] == gold_labels[i] for i in range(total)])
    accuracy = correct / total
    margin = 1.96 * math.sqrt(accuracy * (1 - accuracy) / total)
    
    return accuracy, margin, total


def parse_xlm_logits(logits_file):
    """Parse XLM logits file format (tab-separated with comma-separated logits)."""
    predictions = []
    with open(logits_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            logits = [float(x) for x in parts[1].split(',')]
            pred = int(np.argmax(logits))
            predictions.append(pred)
    return predictions


def parse_hf_predictions(predictions_file):
    """Parse Hugging Face predictions file format."""
    predictions = []
    # XNLI label mapping for string labels
    label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    
    with open(predictions_file) as f:
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
                predictions.append(pred)
    return predictions


def parse_labels(labels_file, task_type="auto"):
    """Parse labels file with automatic format detection."""
    labels = []
    # Label mappings for different tasks
    xnli_label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
    cls_label_map = {'negative': 0, 'positive': 1}
    
    with open(labels_file) as f:
        for line in f:
            label = line.strip()
            if not label:
                continue
                
            # Handle numeric labels
            if label.isdigit():
                labels.append(int(label))
            # Handle string labels
            elif task_type == "xnli" or (task_type == "auto" and label in xnli_label_map):
                labels.append(xnli_label_map.get(label, 0))
            elif task_type == "cls" or (task_type == "auto" and label in cls_label_map):
                labels.append(cls_label_map.get(label, 0))
            else:
                # Default: try to convert to int, otherwise skip
                try:
                    labels.append(int(label))
                except ValueError:
                    print(f"Warning: Unknown label '{label}', skipping...")
                    continue
    return labels


def main():
    parser = argparse.ArgumentParser(description="Unified accuracy calculator for FLUE tasks.")
    parser.add_argument("--predictions_file", type=str, required=True, 
                       help="Path to predictions file (XLM logits or HF predictions)")
    parser.add_argument("--labels_file", type=str, required=True, 
                       help="Path to gold labels file")
    parser.add_argument("--format", type=str, choices=["xlm", "hf", "auto"], default="auto",
                       help="Input format: xlm (logits), hf (Hugging Face), or auto-detect")
    parser.add_argument("--task", type=str, choices=["xnli", "cls", "auto"], default="auto",
                       help="Task type for label mapping: xnli, cls, or auto-detect")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.predictions_file):
        print(f"Error: Predictions file '{args.predictions_file}' not found!")
        exit(1)
    if not os.path.exists(args.labels_file):
        print(f"Error: Labels file '{args.labels_file}' not found!")
        exit(1)
    
    # Auto-detect format if needed
    format_type = args.format
    if format_type == "auto":
        # Check file extension and content to determine format
        if args.predictions_file.endswith('.pred') or 'pred.' in args.predictions_file:
            format_type = "xlm"
        elif 'predict_results' in args.predictions_file:
            format_type = "hf"
        else:
            # Try to detect by looking at first few lines
            try:
                with open(args.predictions_file) as f:
                    first_line = f.readline().strip()
                    if '\t' in first_line and ',' in first_line:
                        format_type = "xlm"
                    else:
                        format_type = "hf"
            except:
                format_type = "hf"  # Default fallback
    
    # Parse predictions based on format
    if format_type == "xlm":
        predictions = parse_xlm_logits(args.predictions_file)
    elif format_type == "hf":
        predictions = parse_hf_predictions(args.predictions_file)
    else:
        print(f"Error: Unknown format '{format_type}'")
        exit(1)
    
    # Parse labels
    labels = parse_labels(args.labels_file, args.task)
    
    # Calculate accuracy
    accuracy, margin, total = calculate_accuracy_with_confidence(predictions, labels)
    
    # Print result
    print(f"Accuracy: {accuracy * 100:.2f}% ± {margin * 100:.2f}% on {total} examples (IC 95%).")


if __name__ == "__main__":
    main()

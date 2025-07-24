#!/usr/bin/env python3
"""
Unified accuracy calculator for FLUE evaluation tasks.
Supports different input formats: XLM logits, Hugging Face predictions, and direct labels.
Automatically creates corrected label files when needed.
"""

import argparse
import csv
import json
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


def parse_hf_eval_results(eval_results_file):
    """Parse Hugging Face eval_results.json file format."""
    try:
        with open(eval_results_file, 'r') as f:
            results = json.load(f)
        
        # Extract accuracy and number of samples
        accuracy = results.get('eval_accuracy', 0.0)
        num_samples = results.get('eval_samples', 0)
        
        if num_samples == 0:
            print("Warning: No evaluation samples found in results file")
            return None, None
        
        print(f"Found Hugging Face evaluation results:")
        print(f"  - Accuracy: {accuracy * 100:.2f}%")
        print(f"  - Samples: {num_samples}")
        
        return accuracy, num_samples
    except Exception as e:
        print(f"Error reading eval_results.json: {e}")
        return None, None


def calculate_confidence_interval(accuracy, num_samples):
    """Calculate 95% confidence interval for accuracy."""
    if num_samples <= 0:
        return 0.0
    
    margin = 1.96 * math.sqrt(accuracy * (1 - accuracy) / num_samples)
    return margin


def find_hf_eval_results(base_path=None):
    """Find Hugging Face eval_results.json file in experiment directories."""
    if base_path is None:
        base_path = "./flue/experiments"
    
    # Common patterns for HF experiment directories
    search_patterns = [
        "**/eval_results.json",
        "flaubert/**/eval_results.json", 
        "cls_hf_*/**/eval_results.json",
        "**/lr_*/**/eval_results.json"
    ]
    
    import glob
    for pattern in search_patterns:
        full_pattern = os.path.join(base_path, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        if matches:
            # Return the most recent one
            return max(matches, key=os.path.getmtime)
    
    return None


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


def create_corrected_labels_from_csv(csv_file_path, output_labels_path):
    """Create corrected labels file from CSV data."""
    print(f"Creating corrected labels from CSV file: {csv_file_path}")
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' not found!")
        return False
    
    labels = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    labels.append(row[1])  # Second column is the label
        
        # Write corrected labels
        with open(output_labels_path, 'w') as f:
            for label in labels:
                f.write(label + '\n')
        
        print(f"Created corrected labels file: {output_labels_path} ({len(labels)} labels)")
        return True
    except Exception as e:
        print(f"Error creating corrected labels: {e}")
        return False


def validate_labels_alignment(labels_file, csv_file_path):
    """Check if labels file is properly aligned with CSV data."""
    if not os.path.exists(labels_file) or not os.path.exists(csv_file_path):
        return False
    
    try:
        # Count lines in labels file
        with open(labels_file, 'r') as f:
            label_lines = sum(1 for line in f if line.strip())
        
        # Count data rows in CSV file (excluding header)
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            csv_lines = sum(1 for row in reader)
        
        return label_lines == csv_lines
    except:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified accuracy calculator for FLUE tasks.",
        epilog="""
Examples:
  # Calculate from XLM predictions
  python3 accuracy_calculator.py --predictions_file model.pred.29 --labels_file test.label --format xlm

  # Calculate from HF predictions
  python3 accuracy_calculator.py --predictions_file predict_results.txt --labels_file test.label --format hf

  # Calculate directly from HF eval_results.json (recommended for HF tasks)
  python3 accuracy_calculator.py --eval_results eval_results.json

  # Auto-detect HF eval_results.json in experiments directory
  python3 accuracy_calculator.py
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions_file", type=str, required=False, 
                       help="Path to predictions file (XLM logits or HF predictions)")
    parser.add_argument("--labels_file", type=str, required=False, 
                       help="Path to gold labels file")
    parser.add_argument("--eval_results", type=str, required=False,
                       help="Path to Hugging Face eval_results.json file")
    parser.add_argument("--format", type=str, choices=["xlm", "hf", "auto"], default="auto",
                       help="Input format: xlm (logits), hf (predictions), or auto-detect")
    parser.add_argument("--task", type=str, choices=["xnli", "cls", "auto"], default="auto",
                       help="Task type for label mapping: xnli, cls, or auto-detect")
    parser.add_argument("--auto_correct", action="store_true", default=True,
                       help="Automatically create corrected labels from CSV if misaligned")
    
    args = parser.parse_args()
    
    # Check for HF eval_results.json format first
    if args.eval_results:
        eval_file = args.eval_results or args.predictions_file
        if not os.path.exists(eval_file):
            print(f"Error: Eval results file '{eval_file}' not found!")
            exit(1)
        
        accuracy, num_samples = parse_hf_eval_results(eval_file)
        if accuracy is not None and num_samples is not None:
            margin = calculate_confidence_interval(accuracy, num_samples)
            print(f"Accuracy: {accuracy * 100:.2f}% ± {margin * 100:.2f}% on {num_samples} examples (IC 95%).")
            return
        else:
            print("Failed to parse eval_results.json file")
            exit(1)
    
    # Auto-detect eval_results.json if no specific file provided
    if not args.predictions_file and not args.eval_results:
        eval_file = find_hf_eval_results()
        if eval_file:
            print(f"Auto-detected HF eval results file: {eval_file}")
            accuracy, num_samples = parse_hf_eval_results(eval_file)
            if accuracy is not None and num_samples is not None:
                margin = calculate_confidence_interval(accuracy, num_samples)
                print(f"Accuracy: {accuracy * 100:.2f}% ± {margin * 100:.2f}% on {num_samples} examples (IC 95%).")
                return
    
    # Fall back to traditional prediction file processing
    if not args.predictions_file:
        print("Error: Either --predictions_file or --eval_results must be provided!")
        exit(1)
    if not args.labels_file:
        print("Error: --labels_file is required for prediction file processing!")
        exit(1)
    
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
                format_type = "¤"  # Default fallback
    
    # Parse predictions based on format
    if format_type == "xlm":
        predictions = parse_xlm_logits(args.predictions_file)
    elif format_type == "hf":
        predictions = parse_hf_predictions(args.predictions_file)
    else:
        print(f"Error: Unknown format '{format_type}'")
        exit(1)
    
    # Parse labels with auto-correction
    labels_file = args.labels_file
    
    # Parse labels
    labels = parse_labels(labels_file, args.task)
    
    # Calculate accuracy
    accuracy, margin, total = calculate_accuracy_with_confidence(predictions, labels)
    
    # Print result
    print(f"Accuracy: {accuracy * 100:.2f}% ± {margin * 100:.2f}% on {total} examples.")


if __name__ == "__main__":
    main()

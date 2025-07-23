#!/usr/bin/env python3
"""
Unified accuracy calculator for FLUE evaluation tasks.
Supports different input formats: XLM logits, Hugging Face predictions, and direct labels.
Automatically creates corrected label files when needed.
"""

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Unified accuracy calculator for FLUE tasks.")
    parser.add_argument("--predictions_file", type=str, required=True, 
                       help="Path to predictions file (XLM logits or HF predictions)")
    parser.add_argument("--labels_file", type=str, required=True, 
                       help="Path to gold labels file")
    parser.add_argument("--format", type=str, choices=["xlm", "hf", "auto"], default="auto",
                       help="Input format: xlm (logits), hf (Hugging Face), or auto-detect")
    parser.add_argument("--task", type=str, choices=["xnli", "cls", "auto"], default="auto",
                       help="Task type for label mapping: xnli, cls, or auto-detect")
    parser.add_argument("--auto_correct", action="store_true", default=True,
                       help="Automatically create corrected labels from CSV if misaligned")
    
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
    
    # Parse labels with auto-correction
    labels_file = args.labels_file
    
    # Auto-correct labels if needed (for HF tasks with CSV data)
    if args.auto_correct and format_type == "hf":
        # Try to find corresponding CSV test file
        base_dir = os.path.dirname(args.labels_file)
        possible_csv_files = [
            os.path.join(base_dir, "test.csv"),
            args.labels_file.replace("test_labels_only.label", "test.csv"),
            args.labels_file.replace(".label", ".csv")
        ]
        
        csv_file = None
        for csv_path in possible_csv_files:
            if os.path.exists(csv_path):
                csv_file = csv_path
                break
        
        if csv_file:
            # Check if current labels file is properly aligned
            if not validate_labels_alignment(args.labels_file, csv_file):
                print(f"Labels file appears misaligned with CSV data ({args.labels_file})")
                print(f"Creating corrected version from CSV: {csv_file}")
                
                # Create corrected labels file
                if "test_labels_only.label" in args.labels_file:
                    corrected_labels_file = args.labels_file.replace("test_labels_only.label", "test_labels_correct.label")
                else:
                    corrected_labels_file = args.labels_file.replace(".label", "_correct.label")
                
                if create_corrected_labels_from_csv(csv_file, corrected_labels_file):
                    labels_file = corrected_labels_file
                    print(f"Using corrected labels file: {labels_file}")
                else:
                    print("Failed to create corrected labels, using original file")
            else:
                print(f"Labels file is properly aligned with CSV data")
    
    # Parse labels
    labels = parse_labels(labels_file, args.task)
    
    # Calculate accuracy
    accuracy, margin, total = calculate_accuracy_with_confidence(predictions, labels)
    
    # Print result
    print(f"Accuracy: {accuracy * 100:.2f}% ± {margin * 100:.2f}% on {total} examples (IC 95%).")


if __name__ == "__main__":
    main()

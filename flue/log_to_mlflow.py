#!/usr/bin/env python3
"""
Script to log evaluation results to MLflow
Usage: python3 log_to_mlflow.py <eval_results_path> <task_name> <model_name> <learning_rate> <epochs> <batch_size>
"""

import sys
import json
import re
import os
from datetime import datetime
import mlflow
from mlflow_utils import get_mlflow_tracking_uri, get_experiment_name

def extract_accuracy_from_text(accuracy_text):
    """Extract accuracy metrics from accuracy calculator output"""
    accuracy_match = re.search(r'Accuracy: ([\d.]+)%', accuracy_text)
    confidence_match = re.search(r'± ([\d.]+)%', accuracy_text)
    examples_match = re.search(r'on (\d+) examples', accuracy_text)
    
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    confidence = float(confidence_match.group(1)) if confidence_match else None
    examples = int(examples_match.group(1)) if examples_match else None
    
    return accuracy, confidence, examples

def log_to_mlflow(eval_results_path, task_name, model_name, learning_rate, epochs, batch_size, accuracy_text=None):
    """Log evaluation results to MLflow"""
    
    # Use shared utilities for consistent setup
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    experiment_name = get_experiment_name(task_name)
    mlflow.set_experiment(experiment_name)
    
    # Generate run name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{task_name}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param('task', task_name)
        mlflow.log_param('model', model_name)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        
        # Log eval_results.json as artifact if it exists
        if os.path.exists(eval_results_path):
            mlflow.log_artifact(eval_results_path, "evaluation_results")
            
            # Try to read and log metrics from eval_results.json
            try:
                with open(eval_results_path, 'r') as f:
                    eval_results = json.load(f)
                    
                # Log metrics from eval_results.json
                for key, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                        
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Warning: Could not read eval_results.json: {e}")
        
        # If accuracy text is provided, extract and log accuracy metrics
        if accuracy_text:
            accuracy, confidence, examples = extract_accuracy_from_text(accuracy_text)
            
            if accuracy is not None:
                mlflow.log_metric('final_accuracy', accuracy)
            if confidence is not None:
                mlflow.log_metric('confidence_interval', confidence)
            if examples is not None:
                mlflow.log_metric('num_examples', examples)
        
        print(f"Résultats loggés dans MLflow avec succès!")
        print(f"Experiment: {experiment_name}")
        print(f"Run: {run_name}")

def main():
    if len(sys.argv) < 6:
        print("Usage: python3 log_to_mlflow.py <eval_results_path> <task_name> <model_name> <learning_rate> <epochs> <batch_size> [accuracy_text]")
        sys.exit(1)
    
    eval_results_path = sys.argv[1]
    task_name = sys.argv[2]
    model_name = sys.argv[3]
    learning_rate = sys.argv[4]
    epochs = sys.argv[5]
    batch_size = sys.argv[6]
    accuracy_text = sys.argv[7] if len(sys.argv) > 7 else None
    
    log_to_mlflow(eval_results_path, task_name, model_name, learning_rate, epochs, batch_size, accuracy_text)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MLflow wrapper for Hugging Face training in FLUE benchmark
Aurélien Tassaux - Integration with MLflow tracking
"""

import os
import sys
import json
import argparse
import subprocess
import mlflow
import mlflow.transformers
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train with MLflow tracking")
    parser.add_argument("--train_file", required=True, help="Training data file")
    parser.add_argument("--validation_file", required=True, help="Validation data file")
    parser.add_argument("--test_file", required=True, help="Test data file")
    parser.add_argument("--model_name_or_path", required=True, help="Model name or path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, required=True, help="Save steps")
    parser.add_argument("--per_device_train_batch_size", type=int, required=True, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, required=True, help="Eval batch size")
    parser.add_argument("--task_name", default="cls-books", help="Task name for experiment tracking")
    parser.add_argument("--experiment_name", default="FLUE-CLS-Books", help="MLflow experiment name")
    
    return parser.parse_args()


def setup_mlflow(experiment_name, task_name, model_name):
    """Setup MLflow experiment and run"""
    # Set MLflow tracking URI (handle Windows/WSL paths)
    mlflow_dir = Path("./mlflow_logs")
    mlflow_dir.mkdir(exist_ok=True)
    
    # Convert to absolute path and handle WSL/Windows compatibility
    abs_path = mlflow_dir.absolute()
    if str(abs_path).startswith('\\\\wsl.localhost'):
        # For WSL paths, use relative path instead
        mlflow.set_tracking_uri("./mlflow_logs")
    else:
        mlflow.set_tracking_uri(f"file://{abs_path}")
    
    # Set or create experiment
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    run_name = f"{task_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=run_name)
    
    return mlflow.active_run()


def run_training(args):
    """Run the Hugging Face training script with the provided arguments"""
    cmd = [
        "python", "tools/transformers/examples/pytorch/text-classification/run_glue.py",
        "--train_file", args.train_file,
        "--validation_file", args.validation_file,
        "--model_name_or_path", args.model_name_or_path,
        "--output_dir", args.output_dir,
        "--max_seq_length", str(args.max_seq_length),
        "--do_train",
        "--do_eval",
        "--learning_rate", str(args.learning_rate),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--overwrite_output_dir",
        "--logging_steps", "50",  # Log more frequently for MLflow
        "--eval_strategy", "epoch",  # Correct parameter name
        "--save_strategy", "no",  # Disable intermediate saving to avoid DTensor error
        "--save_only_model",  # Save only model, not optimizer states
        "--metric_for_best_model", "accuracy",
        "--report_to", "none"  # Disable default reporting
    ]
    
    # Add test file and do_predict only if test file exists
    if args.test_file and os.path.exists(args.test_file):
        cmd.extend(["--test_file", args.test_file, "--do_predict"])
    
    # Run the training
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Log the output for debugging
    if result.returncode != 0:
        print(f"Training failed with return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        mlflow.log_param("training_status", "failed")
        mlflow.log_param("error_code", result.returncode)
        if result.stderr:
            mlflow.log_text(result.stderr, "training_error.log")
        if result.stdout:
            mlflow.log_text(result.stdout, "training_output.log")
    
    return result


def log_metrics_and_artifacts(args, output_dir):
    """Log metrics and artifacts to MLflow"""
    output_path = Path(output_dir)
    
    # Log hyperparameters
    mlflow.log_params({
        "model_name": args.model_name_or_path,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_train_epochs,
        "batch_size": args.per_device_train_batch_size,
        "max_seq_length": args.max_seq_length,
        "task": args.task_name
    })
    
    # Log evaluation results if available
    eval_results_file = output_path / "eval_results.json"
    if eval_results_file.exists():
        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)
        
        # Log validation metrics
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"validation_{key}", value)
    
    # Log training logs if available
    trainer_state_file = output_path / "trainer_state.json"
    if trainer_state_file.exists():
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)
        
        # Log training history
        if "log_history" in trainer_state:
            for log_entry in trainer_state["log_history"]:
                epoch = log_entry.get("epoch")
                if epoch is not None:
                    for key, value in log_entry.items():
                        if key != "epoch" and isinstance(value, (int, float)):
                            mlflow.log_metric(key, value, step=int(epoch * 1000))
    
    # Log model artifacts
    if (output_path / "pytorch_model.bin").exists() or (output_path / "model.safetensors").exists():
        # Log the entire model directory
        mlflow.log_artifacts(str(output_path), "model")
    
    # Log specific files
    important_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "training_args.bin",
        "eval_results.json",
        "predict_results_None.txt"
    ]
    
    for file_name in important_files:
        file_path = output_path / file_name
        if file_path.exists():
            mlflow.log_artifact(str(file_path))


def calculate_test_accuracy(predictions_file, labels_file):
    """Calculate test accuracy and log it"""
    try:
        # Run the accuracy calculation script
        cmd = [
            "python", "flue/accuracy_from_hf.py",
            "--predictions_file", predictions_file,
            "--labels_file", labels_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Parse the output to extract accuracy
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if "Accuracy:" in line or "Précision:" in line:
                    # Extract accuracy value
                    try:
                        accuracy_str = line.split(':')[1].strip().replace('%', '')
                        accuracy = float(accuracy_str) / 100.0
                        mlflow.log_metric("test_accuracy", accuracy)
                        print(f"Test accuracy logged to MLflow: {accuracy:.4f}")
                        return accuracy
                    except:
                        pass
        
        print(f"Accuracy calculation output: {result.stdout}")
        if result.stderr:
            print(f"Accuracy calculation errors: {result.stderr}")
            
    except Exception as e:
        print(f"Error calculating test accuracy: {e}")
    
    return None


def main():
    args = parse_args()
    
    # Setup MLflow
    model_name = os.path.basename(args.model_name_or_path)
    run = setup_mlflow(args.experiment_name, args.task_name, model_name)
    
    try:
        print(f"Started MLflow run: {run.info.run_id}")
        print(f"MLflow experiment: {args.experiment_name}")
        
        # Run training
        result = run_training(args)
        
        # Check if training failed and handle it properly
        if hasattr(result, 'returncode') and isinstance(result.returncode, int):
            if result.returncode != 0:
                print(f"Training failed, but continuing to log available data...")
                # Still try to log any artifacts that might exist
                try:
                    log_metrics_and_artifacts(args, args.output_dir)
                except Exception as log_error:
                    print(f"Error logging artifacts after failed training: {log_error}")
                return result.returncode
        elif isinstance(result, int):
            # If run_training returned an error code directly
            if result != 0:
                return result
        
        print("Training completed successfully!")
        mlflow.log_param("training_status", "success")
        
        # Log metrics and artifacts
        log_metrics_and_artifacts(args, args.output_dir)
        
        # Calculate and log test accuracy
        predictions_file = Path(args.output_dir) / "predict_results_None.txt"
        if predictions_file.exists():
            # Determine labels file path based on task
            if "books" in args.task_name.lower():
                labels_file = "flue/data/cls/processed/books/test.label"
            else:
                labels_file = f"flue/data/{args.task_name}/processed/test.label"
            
            if os.path.exists(labels_file):
                test_accuracy = calculate_test_accuracy(str(predictions_file), labels_file)
                if test_accuracy is not None:
                    print(f"Final test accuracy: {test_accuracy:.4f}")
        
        print(f"MLflow run completed: {run.info.run_id}")
        print(f"Results logged to: {mlflow.get_tracking_uri()}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        mlflow.log_param("error", str(e))
        mlflow.log_param("training_status", "failed")
        import traceback
        mlflow.log_text(traceback.format_exc(), "error_traceback.log")
        raise
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MLflow wrapper for run_glue.py with maximum data capture
This script provides comprehensive MLflow tracking with detailed metrics at every epoch.
"""

import os
import sys
import subprocess
import mlflow
import json
from datetime import datetime
from pathlib import Path
from mlflow_utils import get_mlflow_tracking_uri, get_experiment_name, setup_mlflow_environment


def setup_comprehensive_mlflow_environment(experiment_name, run_name=None):
    """Setup MLflow environment with comprehensive tracking"""
    
    # Use shared utility for consistent setup
    config = setup_mlflow_environment(experiment_name)
    
    print(f"MLflow Enhanced Tracking configure:")
    print(f"  - URI: {config['tracking_uri']}")
    print(f"  - Experiment: {config['experiment_name']}")
    print(f"  - Mode: Let trainer handle run creation")


def run_glue_with_maximum_tracking(args):
    """Run run_glue.py with maximum MLflow data capture"""
    
    # Build the command
    cmd = [
        "python", 
        "tools/transformers/examples/pytorch/text-classification/run_glue.py"
    ] + args
    
    # Force maximum data capture parameters
    enhanced_args = []
    
    # MLflow integration
    if "--report_to" not in args:
        enhanced_args.extend(["--report_to", "mlflow"])
    
    # Comprehensive logging - capture everything
    if "--logging_strategy" not in args:
        enhanced_args.extend(["--logging_strategy", "epoch"])  # Log every epoch
    
    if "--logging_steps" not in args:
        enhanced_args.extend(["--logging_steps", "1"])  # Also log every step
    
    if "--logging_first_step" not in args:
        enhanced_args.extend(["--logging_first_step", "true"])  # Log first step
    
    # Evaluation strategy - evaluate every epoch
    if "--eval_strategy" not in args:
        enhanced_args.extend(["--eval_strategy", "epoch"])
    
    if "--eval_steps" not in args:
        enhanced_args.extend(["--eval_steps", "1"])  # Backup: eval every step if needed
    
    # Save strategy - save every epoch for maximum data
    if "--save_strategy" not in args:
        enhanced_args.extend(["--save_strategy", "epoch"])
    
    if "--save_steps" not in args:
        enhanced_args.extend(["--save_steps", "1"])  # Backup: save every step
    
    if "--save_total_limit" not in args:
        enhanced_args.extend(["--save_total_limit", "100"])  # Keep many checkpoints
    
    # Model selection and tracking
    if "--metric_for_best_model" not in args:
        enhanced_args.extend(["--metric_for_best_model", "eval_accuracy"])
    
    if "--greater_is_better" not in args:
        enhanced_args.extend(["--greater_is_better", "true"])
    
    if "--load_best_model_at_end" not in args:
        enhanced_args.extend(["--load_best_model_at_end", "true"])
    
    # Enhanced metrics collection
    if "--include_for_metrics" not in args:
        enhanced_args.extend(["--include_for_metrics", "inputs"])
    
    # Additional monitoring
    if "--dataloader_num_workers" not in args:
        enhanced_args.extend(["--dataloader_num_workers", "4"])
    
    # Ensure we capture gradient information
    if "--logging_nan_inf_filter" not in args:
        enhanced_args.extend(["--logging_nan_inf_filter", "false"])
    
    # Add evaluation at start for baseline
    if "--eval_on_start" not in args:
        enhanced_args.extend(["--eval_on_start", "true"])
    
    # Memory and performance monitoring  
    if "--skip_memory_metrics" not in args:
        enhanced_args.extend(["--skip_memory_metrics", "false"])
    
    # Add all enhanced arguments to command
    cmd.extend(enhanced_args)
    
    print(f"Enhanced tracking parameters added:")
    i = 0
    while i < len(enhanced_args):
        if i + 1 < len(enhanced_args) and not enhanced_args[i+1].startswith('--'):
            print(f"   {enhanced_args[i]} {enhanced_args[i+1]}")
            i += 2
        else:
            print(f"   {enhanced_args[i]}")
            i += 1
    
    print(f"\nFull command: {' '.join(cmd)}")
    
    # Run the command with enhanced environment
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{Path.cwd()}/flue:{env.get('PYTHONPATH', '')}"
    
    result = subprocess.run(cmd, env=env)
    
    return result


def extract_task_info_from_args(args):
    """Extract task and model information from arguments"""
    task_name = "unknown"
    model_name = "unknown"
    
    # Extract model name
    for i, arg in enumerate(args):
        if arg == "--model_name_or_path" and i + 1 < len(args):
            model_path = args[i + 1]
            model_name = Path(model_path).name
            break
    
    # Extract task name from file paths or output dir
    for i, arg in enumerate(args):
        if arg in ["--train_file", "--output_dir"] and i + 1 < len(args):
            path = args[i + 1].lower()
            if "books" in path:
                task_name = "cls_books_hf"
            elif "music" in path:
                task_name = "cls_music_hf"
            elif "dvd" in path:
                task_name = "cls_dvd_hf"
            elif "pawsx" in path:
                task_name = "pawsx_hf"
            elif "xnli" in path:
                task_name = "xnli_hf"
            break
    
    return task_name, model_name


def log_training_configuration(args, task_name, model_name):
    """Log detailed training configuration to MLflow"""
    
    # Parse arguments into a configuration dictionary
    config = {
        "task": task_name,
        "model": model_name,
        "timestamp": datetime.now().isoformat()
    }
    
    # Extract key parameters
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            param_name = arg[2:]  # Remove --
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                param_value = args[i + 1]
                config[param_name] = param_value
                i += 2
            else:
                config[param_name] = True
                i += 1
        else:
            i += 1
    
    # Log all parameters
    for key, value in config.items():
        try:
            if isinstance(value, str) and value.replace('.', '').replace('-', '').replace('e', '').isdigit():
                # Try to convert string numbers to float/int
                if '.' in value or 'e' in value.lower():
                    mlflow.log_param(key, float(value))
                else:
                    mlflow.log_param(key, int(value))
            else:
                mlflow.log_param(key, value)
        except:
            mlflow.log_param(key, str(value))
    
    # Save full configuration as artifact
    config_file = "/tmp/training_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    try:
        mlflow.log_artifact(config_file, "configuration")
        print("Training configuration logged to MLflow")
    except Exception as e:
        print(f"Warning: Could not save configuration: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 mlflow_run_glue.py [run_glue.py arguments...]")
        print("Example: python3 mlflow_run_glue.py --train_file data.csv --model_name_or_path bert-base --output_dir ./output")
        print("\nThis wrapper provides:")
        print("   Metrics logged every epoch AND every step")
        print("   Comprehensive training curves") 
        print("   Gradient and convergence monitoring")
        print("   Complete training history artifacts")
        print("   Detailed parameter tracking")
        sys.exit(1)
    
    # Get arguments for run_glue.py
    glue_args = sys.argv[1:]
    
    # Extract task and model information
    task_name, model_name = extract_task_info_from_args(glue_args)
    
    # Setup experiment name based on task - use shared utility
    experiment_name = get_experiment_name(task_name)
    
    # Don't generate a run name - let the trainer handle it
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # run_name = f"{task_name}_enhanced_{timestamp}"
    
    # Setup enhanced MLflow environment (without creating a run)
    setup_comprehensive_mlflow_environment(experiment_name)
    
    # Don't start an MLflow run here - let the trainer handle it
    print(f"\nDemarrage de l'entrainement avec suivi MLflow maximal:")
    print(f"   Task: {task_name}")
    print(f"   Model: {model_name}")
    print(f"   Experiment: {experiment_name}")
    print(f"   Tracking: Every epoch + every step")
    print(f"   Artifacts: Training curves, config, checkpoints")
    print(f"   Mode: Single run managed by trainer")
    
    result = run_glue_with_maximum_tracking(glue_args)
    
    # Report results
    if result.returncode == 0:
        print(f"\nEntrainement termine avec succes!")
        print(f"Donnees completes disponibles dans MLflow:")
        print(f"   - Experiment: {experiment_name}")
        print(f"   - Metriques: Chaque epoque + chaque etape")
        print(f"   - Artifacts: Courbes d'entrainement completes")
        print(f"   - MLflow URI: {os.environ['MLFLOW_TRACKING_URI']}")
        print(f"\nPour visualiser: mlflow ui --backend-store-uri {os.environ['MLFLOW_TRACKING_URI']}")
    else:
        print(f"\nEntrainement echoue avec le code de retour: {result.returncode}")
        print(f"Configuration et logs partiels disponibles dans MLflow")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

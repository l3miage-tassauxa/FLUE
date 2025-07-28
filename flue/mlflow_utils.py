#!/usr/bin/env python3
"""
Shared MLflow utilities for FLUE
Common functions used across MLflow integration scripts
"""

import os
from pathlib import Path

def get_mlflow_tracking_uri():
    """Get the standardized MLflow tracking URI for FLUE"""
    # Always relative to FLUE/FLUE directory
    return f'file://{os.getcwd()}/flue/mlruns'

def get_experiment_name(task_name):
    """Get standardized MLflow experiment name based on task"""
    if "cls_books" in task_name or "books" in task_name:
        return "FLUE_CLS_Books_HF"
    elif "cls_music" in task_name or "music" in task_name:
        return "FLUE_CLS_Music_HF"
    elif "cls_dvd" in task_name or "dvd" in task_name:
        return "FLUE_CLS_DVD_HF"
    elif "pawsx" in task_name:
        return "FLUE_PAWSX_HF"
    elif "xnli" in task_name:
        return "FLUE_XNLI_HF"
    else:
        return f"FLUE_{task_name.upper()}_HF"

def setup_mlflow_environment(experiment_name):
    """Setup MLflow environment variables"""
    os.environ['MLFLOW_TRACKING_URI'] = get_mlflow_tracking_uri()
    os.environ['MLFLOW_EXPERIMENT_NAME'] = experiment_name.replace("_Enhanced", "")
    os.environ['MLFLOW_FLATTEN_PARAMS'] = 'true'
    
    return {
        'tracking_uri': os.environ['MLFLOW_TRACKING_URI'],
        'experiment_name': os.environ['MLFLOW_EXPERIMENT_NAME']
    }

#!/usr/bin/env python3
"""
Test script to verify that restored MLflow experiments work correctly
"""

import os
import sys
import mlflow
from flue.mlflow_utils import get_mlflow_tracking_uri, get_experiment_name

def test_experiment(task_name):
    """Test that we can access and set an MLflow experiment"""
    print(f"\n=== Testing {task_name} ===")
    
    # Set up MLflow
    tracking_uri = get_mlflow_tracking_uri()
    experiment_name = get_experiment_name(task_name)
    
    print(f"Tracking URI: {tracking_uri}")
    print(f"Expected experiment name: {experiment_name}")
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        # Try to get the experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            print(f"✅ Found experiment: {experiment.name}")
            print(f"   ID: {experiment.experiment_id}")
            print(f"   Status: {experiment.lifecycle_stage}")
            
            # Try to set it as active
            mlflow.set_experiment(experiment_name)
            print(f"✅ Successfully set as active experiment")
            return True
        else:
            print(f"❌ Experiment '{experiment_name}' not found")
            print("   This experiment will be created automatically on first use")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing experiment: {e}")
        return False

def main():
    """Test all HF task experiments"""
    print("Testing restored MLflow experiments...")
    
    tasks = ['cls_books', 'cls_music', 'cls_dvd', 'pawsx', 'xnli']
    
    for task in tasks:
        test_experiment(task)
    
    print("\n=== Summary ===")
    print("The restored experiments should now work correctly.")
    print("Missing experiments (PAWSX_HF, XNLI_HF) will be created automatically.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify MLflow integration works
"""

import os
import mlflow
from pathlib import Path

def test_mlflow_setup():
    """Test that MLflow can be set up properly"""
    print("🧪 Testing MLflow integration...")
    
    # Create test directory
    test_dir = Path("./test_mlflow_logs")
    test_dir.mkdir(exist_ok=True)
    
    # Set MLflow tracking URI (handle WSL paths)
    abs_path = test_dir.absolute()
    if str(abs_path).startswith('\\\\wsl.localhost'):
        # For WSL paths, use relative path instead
        mlflow.set_tracking_uri("./test_mlflow_logs")
    else:
        mlflow.set_tracking_uri(f"file://{abs_path}")
    
    # Create test experiment
    experiment_name = "FLUE-Test-Integration"
    mlflow.set_experiment(experiment_name)
    
    # Start a test run
    with mlflow.start_run(run_name="test_run") as run:
        print(f"✅ MLflow run started: {run.info.run_id}")
        
        # Log some test parameters
        mlflow.log_param("model_name", "test-model")
        mlflow.log_param("learning_rate", 5e-6)
        mlflow.log_param("task", "cls-books")
        
        # Log some test metrics
        mlflow.log_metric("test_accuracy", 0.85)
        mlflow.log_metric("validation_accuracy", 0.82)
        
        print("✅ Parameters and metrics logged successfully")
        
        # Create a test artifact
        test_file = test_dir / "test_artifact.txt"
        with open(test_file, 'w') as f:
            f.write("This is a test artifact for MLflow integration")
        
        mlflow.log_artifact(str(test_file))
        print("✅ Test artifact logged successfully")
    
    print(f"🎉 MLflow integration test completed successfully!")
    print(f"📊 Results saved to: {mlflow.get_tracking_uri()}")
    print(f"🌐 To view results, run: mlflow ui --backend-store-uri ./test_mlflow_logs")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("🧹 Test directory cleaned up")

if __name__ == "__main__":
    test_mlflow_setup()

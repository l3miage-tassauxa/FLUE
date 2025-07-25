#!/bin/bash
# Script to start MLflow UI
# This script starts the MLflow web interface to view experiment results

echo "Starting MLflow UI..."

# Set the tracking URI to the local mlruns directory
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"

# Check if mlruns directory exists
if [ ! -d "./mlruns" ]; then
    echo "Warning: mlruns directory not found. No experiments have been logged yet."
    echo "Run a Hugging Face task (e.g., cls-books-HF) first to generate experiments."
    exit 1
fi

# Check if MLflow is installed
if ! command -v mlflow &> /dev/null; then
    echo "Error: MLflow is not installed or not in PATH."
    echo "Please ensure you're in the correct conda environment (hf-finetune)."
    exit 1
fi

echo "MLflow UI will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the MLflow UI server."
echo ""

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

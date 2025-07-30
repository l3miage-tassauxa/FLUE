#!/bin/bash
# Script to start MLflow UI
# This script starts the MLflow web interface to view experiment results

echo "Starting MLflow UI..."

# Activate conda environment
source /home/getalp/tassauxa/miniconda3/etc/profile.d/conda.sh
conda activate hf-finetune

# Set the tracking URI to the consolidated mlruns directory in flue/
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
    echo "Installing MLflow in hf-finetune environment..."
    pip install mlflow
fi

# Check if port 5000 is already in use
USER=$(whoami)
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 5000 is already in use. Checking if it's your MLflow process..."
    if ps aux | grep -v grep | grep "$USER.*mlflow.server" > /dev/null; then
        echo "Your MLflow is already running on port 5000."
        echo "Stopping your existing MLflow processes..."
        MLFLOW_PIDS=$(ps aux | grep -v grep | grep "$USER.*mlflow.server" | awk '{print $2}')
        for pid in $MLFLOW_PIDS; do
            kill "$pid"
        done
        sleep 2
        echo "Your existing MLflow processes stopped."
    else
        PORT_OWNER=$(lsof -Pi :5000 -sTCP:LISTEN | tail -n +2 | awk '{print $3}' | head -1)
        echo "Port 5000 is in use by another user ($PORT_OWNER). Trying port 5001..."
        PORT=5001
    fi
else
    PORT=5000
fi

echo "MLflow UI will be available at: http://localhost:${PORT}"
echo "MLflow tracking directory: $(pwd)/mlruns"
echo "Press Ctrl+C to stop the MLflow UI server."
echo ""

# Start MLflow UI with the correct backend store URI
mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 0.0.0.0 --port ${PORT}

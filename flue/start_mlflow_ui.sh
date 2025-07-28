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
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Port 5000 is already in use. Checking if it's MLflow..."
    if ps aux | grep -q "mlflow.server"; then
        echo "MLflow is already running on port 5000."
        echo "Stopping existing MLflow processes..."
        pkill -f "mlflow.server"
        sleep 2
        echo "Existing MLflow processes stopped."
    else
        echo "Another service is using port 5000. Trying port 5001..."
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

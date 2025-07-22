#!/bin/bash
# Script to install MLflow in the hf-finetune conda environment
# Run this script in your WSL terminal

echo "=== Installing MLflow in hf-finetune environment ==="

# Initialize conda for bash if needed
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Initializing conda..."
    eval "$(/home/grimoire/miniconda3/bin/conda shell.bash hook)"
fi

# Source conda and activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hf-finetune

if [ $? -eq 0 ]; then
    echo "✅ Successfully activated hf-finetune environment"
    
    echo "Installing MLflow via pip..."
    pip install mlflow
    
    if [ $? -eq 0 ]; then
        echo "✅ MLflow installed successfully!"
        
        # Verify installation
        echo "Verifying MLflow installation..."
        mlflow --version
        
        if [ $? -eq 0 ]; then
            echo "🎉 MLflow is ready to use!"
            echo ""
            echo "You can now:"
            echo "1. Run experiments with: ./flue/evaluation_auto.sh cls-books-HF true [model_name]"
            echo "2. Start MLflow UI with: mlflow ui --backend-store-uri ./mlflow_logs"
        else
            echo "❌ MLflow installation verification failed"
        fi
    else
        echo "❌ Failed to install MLflow"
    fi
else
    echo "❌ Failed to activate hf-finetune environment"
    echo "Please make sure you have the hf-finetune conda environment created"
fi

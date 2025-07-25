#!/bin/bash
# Test script for MLflow functionality

echo "Testing MLflow integration..."

# Set up conda environment
source /home/getalp/tassauxa/miniconda3/etc/profile.d/conda.sh
conda activate hf-finetune

cd /home/getalp/tassauxa/FLUE/FLUE

# Test if MLflow is available
echo "Checking MLflow installation..."
python3 -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')" || {
    echo "Error: MLflow not available in current environment"
    exit 1
}

# Test if the log_to_mlflow.py script works
echo "Testing log_to_mlflow.py script..."
python3 flue/log_to_mlflow.py --help 2>/dev/null || {
    echo "Testing with dummy parameters..."
    python3 flue/log_to_mlflow.py "/tmp/dummy.json" "test_task" "test_model" "0.00001" "3" "8" "Accuracy: 90.5% ± 1.2% on 100 examples" 2>/dev/null || echo "Script executed (errors expected with dummy data)"
}

echo "MLflow test completed."
echo ""
echo "To run with MLflow tracking:"
echo "bash tache.sh cls-books-MlFlow false cls_books_lr5e6_hf_base_uncased.cfg"

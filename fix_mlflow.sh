#!/bin/bash
# Simple MLflow installation and setup script
# Since you're already in the hf-finetune environment

echo "=== Installing MLflow in current environment ==="

# Install MLflow in the current environment
echo "Installing MLflow..."
pip install --force-reinstall mlflow

if [ $? -eq 0 ]; then
    echo "✅ MLflow installed successfully!"
    
    # Fix permissions for the MLflow executable
    echo "Fixing permissions for MLflow..."
    chmod +x $CONDA_PREFIX/bin/mlflow*
    
    # Verify installation
    echo "Verifying MLflow installation..."
    mlflow --version
    
    if [ $? -eq 0 ]; then
        echo "🎉 MLflow is ready to use!"
        echo ""
        echo "Creating mlflow_logs directory..."
        mkdir -p ./mlflow_logs
        echo ""
        echo "You can now:"
        echo "1. Start MLflow UI with: mlflow ui --backend-store-uri ./mlflow_logs"
        echo "2. Run experiments with: ./flue/evaluation_auto.sh cls-books-HF true [model_name]"
        echo ""
        echo "To start MLflow UI now, run:"
        echo "mlflow ui --backend-store-uri ./mlflow_logs --host 0.0.0.0 --port 5000"
    else
        echo "❌ MLflow installation verification failed"
        echo "Trying to fix permissions manually..."
        find $CONDA_PREFIX/bin -name "*mlflow*" -exec chmod +x {} \;
        echo "Try running: mlflow --version"
    fi
else
    echo "❌ Failed to install MLflow"
fi

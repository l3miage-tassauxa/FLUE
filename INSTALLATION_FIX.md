# 🔧 MLflow Installation Fix for WSL/Linux Environment

## Problem
MLflow was not properly installed in your `hf-finetune` conda environment, causing permission errors when trying to run `mlflow ui`.

## Solution

### Step 1: Install MLflow in your conda environment

In your **WSL/Ubuntu terminal** (not PowerShell), run:

```bash
# Navigate to FLUE directory
cd ~/StageL3/FLUE

# Run the installation script
chmod +x install_mlflow.sh
./install_mlflow.sh
```

### Step 2: Verify installation

```bash
# Activate your environment
conda activate hf-finetune

# Check MLflow is available
mlflow --version
```

### Step 3: Start MLflow UI

```bash
# Make sure you're in the FLUE directory
cd ~/StageL3/FLUE

# Start MLflow UI
./start_mlflow_ui.sh
```

Or manually:
```bash
conda activate hf-finetune
mlflow ui --backend-store-uri ./mlflow_logs
```

## Alternative: Manual Installation

If the script doesn't work, install manually:

```bash
# Activate environment
conda activate hf-finetune

# Install MLflow
pip install mlflow

# Verify
mlflow --version
```

## Running Experiments

Once MLflow is properly installed, you can run experiments:

```bash
# Make sure you're in the correct environment
conda activate hf-finetune

# Navigate to FLUE
cd ~/StageL3/FLUE

# Run an experiment with MLflow tracking
./flue/evaluation_auto.sh cls-books-HF true flaubert_base_cased
```

## Troubleshooting

### If you get "command not found" errors:
```bash
# Refresh your conda installation
conda init bash
source ~/.bashrc
conda activate hf-finetune
```

### If MLflow UI still doesn't work:
```bash
# Check if MLflow is in the correct environment
which mlflow
# Should show something like: /home/grimoire/miniconda3/envs/hf-finetune/bin/mlflow

# If not, reinstall in the environment
pip uninstall mlflow
pip install mlflow
```

### Permission issues:
```bash
# Fix permissions for conda environment
chmod +x /home/grimoire/miniconda3/envs/hf-finetune/bin/*
```

## Ready to Go!

Once everything is working, you should be able to:

1. ✅ Run `mlflow --version` without errors
2. ✅ Start MLflow UI with `./start_mlflow_ui.sh`
3. ✅ Run experiments with automatic MLflow tracking
4. ✅ View results at http://localhost:5000

Let me know if you need any help with these steps!

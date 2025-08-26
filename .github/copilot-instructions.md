# FLUE - French Language Understanding Evaluation

## Project Architecture

FLUE is a comprehensive evaluation framework for French NLP models, similar to GLUE but for French. The project supports both legacy XLM-based evaluations and modern Hugging Face Transformers with MLflow tracking.

### Core Components

- **`flue/evaluation_auto.sh`**: Main orchestration script that handles all evaluation tasks
- **`flue/mlflow_run_glue.py`**: MLflow wrapper around Hugging Face's run_glue.py with comprehensive tracking
- **`flue/mlflow_utils.py`**: Shared utilities for consistent MLflow configuration across all scripts
- **`flue/accuracy_calculator.py`**: Unified accuracy calculation for different output formats (XLM logits, HF predictions)
- **`tache.sh`**: OAR job submission script for cluster execution

### Task Types & Naming Convention

**Hugging Face Tasks** (recommended, with MLflow tracking):
- `cls-books-HF`, `cls-music-HF`, `cls-dvd-HF`: Sentiment classification on Amazon reviews
- `xnli-HF`: Cross-lingual natural language inference 
- `pawsx-HF`: Paraphrase detection

**XLM Tasks** (legacy):
- `cls-books-XLM`, `xnli-XLM`, etc.: Original XLM framework implementations

### MLflow Integration Pattern

All `-HF` tasks use unified experiment naming:
- `FLUE_CLS_Books_HF`, `FLUE_CLS_Music_HF`, `FLUE_CLS_DVD_HF`
- `FLUE_XNLI_HF`, `FLUE_PAWSX_HF`

MLflow directory: `flue/mlruns/` (consolidated from scattered locations)

## Development Workflows

### Running Evaluations

```bash
# Interactive execution
bash ./flue/evaluation_auto.sh cls-books-HF false config_file.cfg

# Cluster submission (OAR)
bash tache.sh cls-books-HF false config_file.cfg
```

### MLflow UI Management

```bash
# Start MLflow dashboard (user-specific for multi-user safety)
bash flue/start_mlflow_ui.sh

# Stop MLflow processes
bash flue/stop_mlflow_ui.sh
```

### Data Preparation Pipeline

1. Download data: `bash flue/get-data-{task}.sh $DATA_DIR`
2. Process data: `bash flue/prepare-data-{task}.sh $DATA_DIR $MODEL_DIR $do_lower`
3. Extract features: `python flue/extract_{task}.py`

## Project-Specific Conventions

### Configuration System

Config files in `flue/examples/` define model paths and hyperparameters:
```bash
exp_name="cls_books_xlm_base_cased"
model_path="./flue/pretrained_models/flaubert_base_cased_xlm_books"
lre=5e-6  # Encoder learning rate
lrp=5e-6  # Projection learning rate
```

### Directory Structure Requirements

- Models: `flue/pretrained_models/{model_name}/` containing `*.pth`, `codes`, `vocab`
- Data: `flue/data/{task}/raw/` and `flue/data/{task}/processed/`
- Experiments: `flue/experiments/{task}_{framework}_{model}/`
- Must run from `FLUE/FLUE/` directory (script validates `$(basename "$PWD")` == "FLUE")

### Error Handling Patterns

Scripts validate:
- Required arguments count
- File existence before processing
- Conda environment activation (`hf-finetune` for HF tasks, `XLM` for legacy)
- Data download completion

### Multi-User Server Safety

MLflow UI scripts filter processes by `$(whoami)` to prevent interference:
```bash
MLFLOW_PIDS=$(ps aux | grep -v grep | grep "$USER.*mlflow.server" | awk '{print $2}')
```

## Integration Points

### Hugging Face Transformers Integration

`mlflow_run_glue.py` wraps `tools/transformers/examples/pytorch/text-classification/run_glue.py` with:
- Enhanced logging every step (`--logging_steps 1`)
- Evaluation every epoch (`--eval_strategy epoch`)
- Comprehensive metric collection (`--include_for_metrics ["input_ids", "attention_mask", "labels"]`)

### Cross-Component Communication

- Shared MLflow configuration via `mlflow_utils.py`
- Standardized experiment naming across all HF tasks
- Unified accuracy calculation supporting both XLM logits and HF JSON outputs
- Clean terminal output (no emoji symbols for professional appearance)

## Key Files for AI Understanding

- `flue/evaluation_auto.sh`: Main task orchestration logic with English messages and comments
- `flue/mlflow_run_glue.py`: Modern HF integration with maximum data capture
- `flue/README.md`: Complete English documentation for the evaluation framework
- `tache.sh`: Cluster execution patterns and conda environment management
- `flue/mlflow_utils.py`: Shared configuration patterns

## Recent Improvements

- **Complete English Translation**: All French echo messages, comments, and documentation have been translated to English for better international accessibility
- **Professional Terminal Output**: Clean terminal output without emoji symbols for professional usage
- **Comprehensive Documentation**: Detailed English README with examples, troubleshooting, and configuration guides

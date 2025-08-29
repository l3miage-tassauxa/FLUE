# FLUE Evaluation Framework

FLUE (French Language Understanding Evaluation) is a comprehensive evaluation framework for French language models. This guide explains how to use the `evaluation_auto.sh` script to evaluate your models on various French NLP tasks.

## Quick Start

1. **Clone the repository**
2. **Ensure you're in the FLUE root directory**
3. **Run evaluation**:
   ```bash
   bash ./flue/evaluation_auto.sh <task> <install_libs> <config_file>
   ```

## Usage

### Command Structure

```bash
bash ./flue/evaluation_auto.sh <task> <install_libs> <config_file>
```

**All three parameters are required:**
- `<task>`: The evaluation task to run
- `<install_libs>`: Whether to install dependencies (`true`/`false`)
- `<config_file>`: Configuration file name (must exist in `flue/examples/`)

### Get Help

```bash
bash ./flue/evaluation_auto.sh --help
# or
bash ./flue/evaluation_auto.sh -h
```

### Available Tasks

#### XLM-based Tasks
- `cls-books-XLM` : Sentiment classification - books
- `cls-music-XLM` : Sentiment classification - music  
- `cls-dvd-XLM` : Sentiment classification - DVD
- `xnli-XLM` : Cross-lingual natural language inference
- `pawsx-XLM` : Paraphrase identification

#### Hugging Face Tasks
- `cls-books-HF` : Books classification - HF
- `cls-music-HF` : Music classification - HF
- `cls-dvd-HF` : DVD classification - HF
- `pawsx-HF` : PAWSX with Hugging Face

#### MLflow Tasks
- `mlflow-cls-books-HF` : Books classification with MLflow tracking
- `mlflow-cls-music-HF` : Music classification with MLflow tracking
- `mlflow-cls-dvd-HF` : DVD classification with MLflow tracking
- `mlflow-pawsx-HF` : PAWSX with MLflow tracking

#### Unimplemented Tasks
- `xnli-HF` : XNLI with Hugging Face integration (to be implemented)
- `parsing` : Syntactic parsing (to be implemented)
- `wsd` : Word sense disambiguation (to be implemented)

## Examples

### Sentiment Classification (Books)
```bash
bash ./flue/evaluation_auto.sh cls-books-XLM true cls_books_lr5e6_xlm_base_cased.cfg
```

### XNLI with Dependencies Installation
```bash
bash ./flue/evaluation_auto.sh xnli-XLM true xnli_config_xlm_base_cased.cfg
```

### Hugging Face Task
```bash
bash ./flue/evaluation_auto.sh cls-books-HF false cls_books_lr5e6_hf.cfg
```

### PAWSX Task
```bash
bash ./flue/evaluation_auto.sh pawsx-HF true pawsx_config_hf.cfg
```

## Configuration

### Available Configuration Files

The framework includes configurations in `flue/examples/`:
- `cls_books_lr5e6_xlm_base_cased.cfg` - Books classification (XLM)
- `cls_books_lr5e6_hf.cfg` - Books classification (HF)
- Other task-specific configurations

### Configuration File Structure

```bash
# Model parameters
model_type=flaubert
model_name=flaubert_base_cased
model_name_or_path=flue/pretrained_models/flaubert_base_cased

# Training parameters
batch_size=8
lr=0.000005
epochs=10
dropout=0.1

# Data paths
data_dir=flue/data/xnli/processed-csv
train_file=flue/data/xnli/processed-csv/train.csv
validation_file=flue/data/xnli/processed-csv/valid.csv
test_file=flue/data/xnli/processed-csv/test.csv
```

## Data Requirements

Tasks require different datasets:

### Classification Tasks (CLS)
- **Location**: `flue/data/cls/`
- **Required file**: `cls-acl10-unprocessed.tar.gz`
- **Download**: Available from Zenodo (see main documentation)

### XNLI Tasks
- **Location**: `flue/data/xnli/`
- **Required files**: French-translated XNLI data
- **Format**: Processed CSV files

### PAWSX Tasks
- **Location**: `flue/data/pawsx/`
- **Required files**: French PAWSX data

### Parsing/WSD Tasks
- **Status**: Not implemented (under development)

## Results

### Output Location

Results are saved in: `flue/experiments/<model_type>/<exp_name>/<exp_id>/`

### Result Files

- `eval_results.json`: Validation accuracy and metrics
- `predict_results_None.txt`: Test predictions
- `training_logs/`: Training progress logs
- Model checkpoints (if enabled)

### Accuracy Calculation

The framework automatically calculates and displays:
- **Validation accuracy** from training logs
- **Test accuracy** from predictions vs. ground truth labels

## Troubleshooting

### Common Issues

1. **Insufficient arguments**
   ```
   Usage: bash ./flue/evaluation_auto.sh <task> <install_libs> <config_file>
   ```
   **Solution**: Provide all three required parameters

2. **Invalid task**
   ```
   Please specify a valid task.
   ```
   **Solution**: Use `--help` to see available tasks

3. **Configuration file not found**
   ```
   Configuration file 'my_config.cfg' not found
   ```
   **Solution**: Ensure the file exists in `flue/examples/`

4. **Data not found (CLS)**
   ```
   Error: cls-acl10-unprocessed.tar.gz not found
   ```
   **Solution**: Download CLS data from Zenodo

5. **Permission denied**
   ```
   Error: Permission denied
   ```
   **Solution**: Run `chmod +x ./flue/evaluation_auto.sh`

6. **Unimplemented tasks**
   ```
   task not yet implemented...
   ```
   **Solution**: These tasks are under development

### Dependencies

Install required libraries by setting the second parameter to `true`:
```bash
bash ./flue/evaluation_auto.sh my_task true my_config.cfg
```

The script will automatically install:
- XLM dependencies (for XLM tasks)
- Hugging Face dependencies (for HF tasks)

## Contributing

### Adding a New Task

To implement a new task in `evaluation_auto.sh`:

1. **Add the case in the switch**:
```bash
"my_new_task")
    echo "Running my new task..."
    # Your implementation code here
    ;;
```

2. **Add to help**:
```bash
# In show_usage() function, add:
echo "  my_new_task    : Description of my task"
```

3. **Test implementation**:
```bash
bash ./flue/evaluation_auto.sh my_new_task false test_config.cfg
```

### Contributing Guidelines

- Keep consistency with existing tasks
- Add appropriate error messages
- Document new tasks in README files
- Test with different configurations

### File Structure

- `evaluation_auto.sh` : Main evaluation script
- `flue/examples/` : Configuration files
- `flue/data/` : Evaluation datasets
- Documentation in README files

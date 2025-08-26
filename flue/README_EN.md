# FLUE Evaluation Framework

FLUE (French Language Understanding Evaluation) is a comprehensive evaluation framework for French language models. This guide explains how to use the `evaluation_auto.sh` script to evaluate your models on various French NLP tasks.

## Quick Start

1. **Clone the repository**
2. **Place your model** in `flue/pretrained_models/your_model_name/`
3. **Run evaluation**:
   ```bash
   bash ./flue/evaluation_auto.sh <task> <install_libs> [model_name] [config_file]
   ```

## Usage

### Basic Command Structure

```bash
bash ./flue/evaluation_auto.sh <task> <install_libs> [model_name] [config_file]
```

**Parameters:**
- `<task>`: Required. The evaluation task to run
- `<install_libs>`: Required. Whether to install dependencies (`true`/`false`)
- `[model_name]`: Optional. Name of your model directory (default: `flaubert_base_cased`)
- `[config_file]`: Optional. Path to custom configuration file

### Available Tasks

#### Hugging Face Tasks (Recommended)
- **`cls-HF`**: Cross-lingual sentiment analysis with Hugging Face Transformers
- **`xnli-HF`**: Cross-lingual natural language inference with Hugging Face Transformers

#### XLM Tasks (Legacy)
- **`cls-XLM`**: Cross-lingual sentiment analysis with XLM framework
- **`xnli-XLM`**: Cross-lingual natural language inference with XLM framework
- **`pawsx`**: Paraphrase Adversaries from Word Scrambling for Cross-lingual Understanding

## Examples

### 1. Evaluate with Default Model

```bash
# Use default flaubert_base_cased model, install libraries
bash ./flue/evaluation_auto.sh xnli-HF true

# Use default model, skip library installation
bash ./flue/evaluation_auto.sh cls-HF false
```

### 2. Evaluate with Your Own Model

```bash
# Evaluate your custom model
bash ./flue/evaluation_auto.sh xnli-HF true my_french_model

# Evaluate CamemBERT
bash ./flue/evaluation_auto.sh cls-HF false camembert-base
```

### 3. Use Custom Configuration

```bash
# Use your own configuration file
bash ./flue/evaluation_auto.sh xnli-HF true my_model path/to/my_config.cfg
```

## Model Configuration

### Directory Structure

Place your models in the `flue/pretrained_models/` directory:

```
flue/pretrained_models/
├── flaubert_base_cased/          # Default model
├── my_french_model/              # Your custom model
├── camembert-base/               # CamemBERT
└── your_model_name/              # Any other model
    ├── config.json
    ├── pytorch_model.bin (or model.safetensors)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

### Supported Model Types

- **FlauBERT**: `flaubert_base_cased`, `flaubert_base_uncased`
- **CamemBERT**: `camembert-base`, `camembert-large`
- **Custom models**: Any Hugging Face compatible French model
- **Fine-tuned models**: Your own fine-tuned versions

## Configuration Files

### Default Configurations

The framework includes default configurations in `flue/examples/`:
- `xnli_lr5e6_hf_base_uncased.cfg` - Default XNLI configuration
- `cls_books_lr5e6_hf_base_uncased.cfg` - Default CLS configuration
- `xnli_lr5e6_xlm_base_cased.cfg` - XLM XNLI configuration
- `pawsx_lr5e6_xlm_base_cased.cfg` - PAWSX configuration

### Custom Configuration

Create your own `.cfg` file with these parameters:

```bash
# Model parameters
model_type=flaubert
model_name=my_model
model_name_or_path=flue/pretrained_models/my_model

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

# Output
output_dir=flue/experiments/my_model/results
max_seq_length=512
```

## Data Requirements

### XNLI Task
- **Automatic**: Data is downloaded automatically from Facebook's XNLI dataset
- **No manual configuration required**

### CLS Task
1. **Manual download required**: Visit [https://zenodo.org/record/3251672](https://zenodo.org/record/3251672)
2. **Request access** to the CLS dataset
3. **Place the file**: `cls-acl10-unprocessed.tar.gz` in `flue/data/cls/raw/`

### PAWSX Task
- **Automatic**: Data is downloaded automatically
- **No manual configuration required**

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

1. **Model not found**
   ```
   Error: Model 'my_model' not found in flue/pretrained_models/
   ```
   **Solution**: Ensure your model directory exists with all required files

2. **Data not found (CLS)**
   ```
   Error: cls-acl10-unprocessed.tar.gz not found
   ```
   **Solution**: Download CLS data from Zenodo (see Data Requirements)

3. **GPU memory issues**
   - Reduce `batch_size` in your configuration file
   - Use a smaller model
   - Reduce `max_seq_length`

4. **Permission denied**
   ```
   Error: Permission denied
   ```
   **Solution**: Run `chmod +x ./flue/evaluation_auto.sh`

5. **Missing parameter**
   ```
   Please specify whether libraries should be installed (true/false).
   ```
   **Solution**: The script now validates parameters for each task - ensure you provide all required arguments

### Dependencies

Install required libraries by setting the second parameter to `true`:

```bash
bash ./flue/evaluation_auto.sh xnli-HF true
```

This installs:
- transformers
- datasets
- torch
- pandas
- scikit-learn
- Other dependencies from `libraries/hg-requirements.txt`

## Advanced Usage

### Environment Variables

You can override configuration parameters via environment variables:
```bash
export MODEL_NAME=my_custom_model
export BATCH_SIZE=16
bash ./flue/evaluation_auto.sh xnli-HF false
```

### Custom Evaluation Metrics

Add your own evaluation scripts following the pattern of:
- `flue/accuracy_from_hf.py` - Hugging Face results processing
- `flue/accuracy_from_task3.py` - XLM results processing

### Modular Argument Validation

The script now uses a modular approach for argument validation:
- Each task validates its own required parameters
- `INSTALL_LIBS` validation is done at the task level
- This improves maintainability and code clarity

## Contributing

To add new tasks or models:
1. Create configuration files in `flue/examples/`
2. Add case handling in `evaluation_auto.sh`
3. Implement data preprocessing if needed
4. Add result processing scripts

## License

This framework is based on the original FLUE benchmark. Please cite the original paper when using this evaluation framework.

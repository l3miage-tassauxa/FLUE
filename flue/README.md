# FLUE Evaluation Framework

FLUE (French Language Understanding Evaluation) is a comprehensive evaluation framework for French language models. This guide explains how to use the `evaluation_auto.sh` script to evaluate your models on various French NLP tasks.

## Quick Start

1. **Clone the repository**
2. **Place your model** in `flue/pretrained_models/your_model_name/` (if using custom models)
3. **Run evaluation**:
   ```bash
   bash ./flue/evaluation_auto.sh <task> <install_libs> [config_file]
   ```

## Usage

### Basic Command Structure

```bash
bash ./flue/evaluation_auto.sh <task> <install_libs> [config_file]
```

**Parameters:**
- `<task>`: Required. The evaluation task to run
- `<install_libs>`: Required. Whether to install dependencies (`true`/`false`)
- `[config_file]`: Optional. Path to custom configuration file (not model_name)

### Available Tasks

#### Hugging Face Tasks (Recommended)
- **`cls-books-HF`**: Cross-lingual sentiment analysis on books domain with Hugging Face Transformers
- **`cls-music-HF`**: Cross-lingual sentiment analysis on music domain with Hugging Face Transformers  
- **`cls-dvd-HF`**: Cross-lingual sentiment analysis on DVD domain with Hugging Face Transformers
- **`xnli-HF`**: Cross-lingual natural language inference with Hugging Face Transformers
- **`pawsx-HF`**: Paraphrase identification with Hugging Face Transformers

#### XLM Tasks (Legacy)
- **`cls-books-XLM`**: Cross-lingual sentiment analysis on books domain with XLM framework
- **`cls-music-XLM`**: Cross-lingual sentiment analysis on music domain with XLM framework
- **`cls-dvd-XLM`**: Cross-lingual sentiment analysis on DVD domain with XLM framework
- **`xnli-XLM`**: Cross-lingual natural language inference with XLM framework
- **`pawsx-XLM`**: Paraphrase identification with XLM framework

#### Additional Tasks
- **`parse`**: Constituency parsing (requires FrenchTreeBank dataset)
- **`wsd`**: Word Sense Disambiguation (requires FrenchSemEval dataset)

## Examples

### 1. Evaluate with Default Model

```bash
# Use default model, install libraries for XNLI task
bash ./flue/evaluation_auto.sh xnli-HF true

# Use default model, skip library installation for sentiment analysis
bash ./flue/evaluation_auto.sh cls-books-HF false
```

### 2. Evaluate with Custom Configuration

```bash
# Evaluate with custom configuration file
bash ./flue/evaluation_auto.sh cls-music-HF true my_config.cfg

# Evaluate DVD sentiment analysis with custom config
bash ./flue/evaluation_auto.sh cls-dvd-HF false custom_dvd_config.cfg
```

### 3. Multiple Domain Evaluation

```bash
# Evaluate all CLS domains with Hugging Face
bash ./flue/evaluation_auto.sh cls-books-HF true
bash ./flue/evaluation_auto.sh cls-music-HF false  
bash ./flue/evaluation_auto.sh cls-dvd-HF false
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

**Hugging Face Configurations:**
- `cls_books_lr5e6_hf_base_uncased.cfg` - CLS books with Hugging Face
- `cls_music_lr5e6_hf_base_uncased.cfg` - CLS music with Hugging Face  
- `cls_dvd_lr5e6_hf_base_uncased.cfg` - CLS DVD with Hugging Face
- `pawsx_lr5e6_hf_base_cased.cfg` - PAWSX with Hugging Face

**XLM Configurations:**
- `cls_books_lr5e6_xlm_base_cased.cfg` - CLS books with XLM
- `cls_music_lr5e6_xlm_base_cased.cfg` - CLS music with XLM
- `cls_dvd_lr5e6_xlm_base_cased.cfg` - CLS DVD with XLM
- `xnli_lr5e6_xlm_base_cased.cfg` - XNLI with XLM
- `pawsx_lr5e6_xlm_base_cased.cfg` - PAWSX with XLM

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

### CLS Tasks (books, music, dvd)
1. **Manual download required**: Visit [https://zenodo.org/record/3251672](https://zenodo.org/record/3251672)
2. **Request access** to the CLS dataset
3. **Place the file**: `cls-acl10-unprocessed.tar.gz` in `flue/data/cls/raw/`

### PAWSX Task
- **Automatic**: Data is downloaded automatically
- **No manual configuration required**

### Parse Task
1. **Manual download required**: Visit [FrenchTreeBank dataset page]
2. **Download and extract** the FrenchTreeBank dataset
3. **Place in**: `flue/data/parse/` directory

### WSD (Word Sense Disambiguation) Task
1. **Manual download required**: Visit [http://www.llf.cnrs.fr/dataset/fse/](http://www.llf.cnrs.fr/dataset/fse/)
2. **Download** the FrenchSemEval (FSE) dataset
3. **Extract to**: `flue/data/wsd/FSE-1.1-10_12_19/` directory

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

1. **Configuration file not found**
   ```
   Error: configuration file 'my_config.cfg' does not exist in flue/examples directory.
   ```
   **Solution**: Ensure your configuration file exists in the `flue/examples/` directory

2. **CLS data not found**
   ```
   You need to request access to the data at https://zenodo.org/record/3251672
   ```
   **Solution**: Download CLS data from Zenodo and place in `flue/data/cls/raw/`

3. **WSD data not found**
   ```
   Error: WSD data is not available in flue/data/wsd/
   ```
   **Solution**: Download FrenchSemEval dataset and extract to `flue/data/wsd/FSE-1.1-10_12_19/`

4. **Missing parameters**
   ```
   Please specify whether libraries should be installed (true/false).
   ```
   **Solution**: Provide all required parameters: `<task>` and `<install_libs>`

5. **Invalid task specified**
   ```
   Please specify a valid task.
   ```
   **Solution**: Use one of the valid tasks listed in the Available Tasks section

6. **GPU memory issues**
   - Reduce `batch_size` in your configuration file
   - Use a smaller model
   - Reduce `max_seq_length`

7. **Permission denied**
   ```
   Error: Permission denied
   ```
   **Solution**: Run `chmod +x ./flue/evaluation_auto.sh`

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

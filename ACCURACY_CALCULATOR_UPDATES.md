# FLUE Accuracy Calculator - Updated Features

## Summary of Changes

The `accuracy_calculator.py` script has been enhanced to support Hugging Face `eval_results.json` files directly, providing more accurate results with confidence intervals. Additionally, XLM tasks now use dynamic epoch calculation from config files.

## New Features

### 1. Direct HF eval_results.json Support
- **Input**: Hugging Face `eval_results.json` files
- **Output**: Accuracy with 95% confidence interval
- **Advantage**: Uses the exact accuracy and sample count from HF evaluation

### 2. Auto-Detection
- Automatically finds `eval_results.json` files in experiment directories
- Auto-finds latest XLM prediction files in specified directories
- No need to specify file paths manually

### 3. Dynamic Epoch Calculation for XLM Tasks
- **evaluation_auto.sh** now uses `$((num_epochs - 1))` instead of hardcoded "29"
- Automatically adapts to different epoch configurations from config files
- More flexible and maintainable

### 4. Confidence Intervals
- Calculates 95% confidence intervals for all accuracy measurements
- More statistically robust than simple accuracy percentages

## Usage Examples

### For HF Tasks (Recommended)
```bash
# Use specific eval_results.json file
python3 flue/accuracy_calculator.py --eval_results path/to/eval_results.json

# Auto-detect from experiments directory
python3 flue/accuracy_calculator.py
```

### For XLM Tasks (Enhanced)
```bash
# Traditional approach
python3 flue/accuracy_calculator.py --predictions_file model.pred.29 --labels_file test.label --format xlm

# Auto-find latest prediction file
python3 flue/accuracy_calculator.py --auto_find_pred experiments/model_dir/ --labels_file test.label

# Auto-find specific epoch
python3 flue/accuracy_calculator.py --auto_find_pred experiments/model_dir/ --target_epoch 29 --labels_file test.label
```

## Integration with evaluation_auto.sh

The evaluation script has been updated to:
- **XLM Tasks**: Use `$((num_epochs - 1))` for prediction file paths
- **HF Tasks**: Use `--eval_results` for JSON-based accuracy calculation
- More consistent output format across all tasks
- Better error handling and margin of error calculations

## Benefits

1. **Adaptability**: XLM tasks automatically adapt to different epoch configurations
2. **Accuracy**: Direct access to HF evaluation metrics
3. **Consistency**: Same confidence interval calculation for all tasks  
4. **Flexibility**: Auto-detection and smart file finding
5. **Reliability**: Better error handling and validation

## Example Output

### XLM Task with Auto-Detection:
```
Found target prediction file: test.pred.29
Auto-detected prediction file: ./experiments/cls_dvd_xlm_base_cased/.../test.pred.29
Accuracy: 91.65% ± 1.21% on 1999 examples.
```

### HF Task with eval_results.json:
```
Found Hugging Face evaluation results:
  - Accuracy: 97.49%
  - Samples: 399
Accuracy: 97.49% ± 1.53% on 399 examples (IC 95%).
```

This provides both the accuracy and the margin of error, giving you a complete statistical picture of your model's performance while being adaptable to different configurations.

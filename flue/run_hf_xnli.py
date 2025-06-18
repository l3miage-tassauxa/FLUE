from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd

# Replace with your actual model name or path
model_name = "./flue/pretrained_model/Text_Base_fr_4GB_v0/checkpoint_best.pt"  # e.g. "flaubert/flaubert_base_cased" or local path

# Load your TSV files
def load_tsv(file):
    df = pd.read_csv(file, sep='\t', header=None, names=["text", "label"])
    # Split text into premise and hypothesis if needed
    df[["premise", "hypothesis"]] = df["text"].str.split('\t', expand=True)
    return Dataset.from_pandas(df[["premise", "hypothesis", "label"]])

data_files = {
    "train": "flue/data/xnli/processed/train.tsv",
    "validation": "flue/data/xnli/processed/valid.tsv",
    "test": "flue/data/xnli/processed/test.tsv"
}
raw_datasets = DatasetDict({k: load_tsv(v) for k, v in data_files.items()})

# Label mapping if needed
label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
def encode_labels(example):
    example["label"] = label2id.get(example["label"], 0)
    return example

raw_datasets = raw_datasets.map(encode_labels)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize
def preprocess(example):
    return tokenizer(example["premise"], example["hypothesis"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = raw_datasets.map(preprocess, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate(tokenized_datasets["test"])
print(results)
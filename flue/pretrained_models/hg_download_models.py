from transformers import AutoModel, AutoTokenizer, FlaubertModel, FlaubertTokenizer

model_name = "flaubert/flaubert_base_cased"
cache_dir = "./flue/pretrained_models"

# Download and cache the model
model, log = AutoModel.from_pretrained(model_name, output_loading_info=True, cache_dir=cache_dir)
# flaubert, log = FlaubertModel.from_pretrained(model_name, output_loading_info=True, cache_dir=cache_dir)

# Download and cache the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# flaubert_tokenizer = FlaubertTokenizer.from_pretrained(model_name, do_lowercase=False, cache_dir=cache_dir)
# For Flaubert models, do_lowercase=False if using cased models, True if using uncased ones

print(f"Model and tokenizer downloaded to {cache_dir}")
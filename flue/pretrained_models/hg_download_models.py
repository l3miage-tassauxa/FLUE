from transformers import AutoModel, AutoTokenizer, FlaubertModel, FlaubertTokenizer

model_name = "flaubert/flaubert_base_cased"
cache_dir = "./flue/pretrained_models"

# Download and cache the model
model, log = AutoModel.from_pretrained(model_name, output_loading_info=True, cache_dir=cache_dir)
# flaubert, log = FlaubertModel.from_pretrained(model_name, output_loading_info=True, cache_dir=cache_dir)

# Log the loading information
# print("\nInformations de chargement du modèle :")
# print(f"Clés manquantes (missing_keys) : {log['missing_keys']}")
# print(f"Clés inattendues (unexpected_keys) : {log['unexpected_keys']}")
# print(f"Clés non correspondantes (mismatched_keys) : {log['mismatched_keys']}")


# Download and cache the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
# flaubert_tokenizer = FlaubertTokenizer.from_pretrained(model_name, do_lowercase=False, cache_dir=cache_dir)
# do_lowercase=False if using cased models, True if using uncased ones

print(f"Model and tokenizer downloaded to {cache_dir}")
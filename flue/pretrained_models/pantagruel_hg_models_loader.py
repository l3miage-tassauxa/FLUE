import transformers

model = transformers.AutoModel.from_pretrained('flue/pretrained_models/Text_Base_fr_4GB_v0', trust_remote_code=True, local_files_only=True)

tokenizer = transformers.AutoTokenizer.from_pretrained('flue/pretrained_models/Text_Base_fr_4GB_v0', trust_remote_code=True, local_files_only=True)
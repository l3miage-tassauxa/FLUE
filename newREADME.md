
This repository is **still under construction**. 

This repository shares: pre-trained models (base and large), the data, and the code to use the models, and Fine-tuning them on the FLUE benchmark.

TODO: Quick presentation of 'Text_Base' and Speech_Base' models (uni-modal speech and text SSL models for French)

[**FLUE**](https://github.com/getalp/Flaubert/tree/master/flue): an evaluation setup for French NLP systems similar to the popular GLUE benchmark. The goal is to enable further reproducible experiments in the future and to share models and progress on the French language. 

# Table of Contents
**1. [Install dependencies for Flaubert Models](#1-flaubert-models)**  

The pretrained models are available for download from [here](https://zenodo.org/records/13883578) 

TODO: complete the table by replacing the 'x' with the correct values

| Model name | Number of layers | Attention Heads | Embedding Dimension | Total Parameters |
| :------:       |   :---: | :---: | :---: | :---: |
| `Text_Base_fr_4GB_v0` | x    | x    | x   | x M |
| `Text_Base_fr_4GB_v1`  | x  | x  | x  | x M |
| `Speech_Base_en_1K`   | x   | x      | x   | x M |
| `Speech_Base_fr_1K`  | x   | x     | x | x M |
| `Speech_Large_fr_14K`  | x   | x     | x | x M |
| `Speech_Large_fr_14K_v1`  | x   | x     | x | x M |


other models:
- Speech_Base_fr_14K
- Speech_Base_fr_14K_prenet6L
- Speech_Base_fr_1K_prenet6L
- Text_Base_fr_4GB_camtok_step500K
- Text_Base_fr_4GB_camtok_step1M
- Text_Base_fr_4GB_camtok_step2M
- Text_Base_fr_OSCAR_camtok

are provided [here](https://zenodo.org/uploads/15101370?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQwMjAxM2FkLTJlMDUtNDlhOS05MjhiLWVhOGZjYjBhZjQzOCIsImRhdGEiOnt9LCJyYW5kb20iOiJkMGViMWJhNDc2YTI0ZWFmOWFlYjJmMDE0ZDI1NjIyYyJ9.nNs2v7H4FdYYprSM9qnX1mjMPWRZSjjBwwe1TCp-zBiFFxi_VFjO-bYxXKgSTojTnLeB5Y2kBr8PJxCz9iqtmw)

# 1. FlauBERT models

/!\ Comme le repository de [Flaubert](https://github.com/getalp/Flaubert) est ancien, la mise en place d'un environnement python (ici avec conda) adapté est nécessaire, comprenant 


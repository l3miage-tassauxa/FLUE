#!/bin/bash

# NB: This script is intended to be run in the 'FLUE' repository.

# Prompt user for the model download link
read -p "Enter the download link for the language model (.tar.gz): " MODEL_URL

# Extract the filename and model name
FILENAME=$(basename "$MODEL_URL")
MODEL_NAME="${FILENAME%.tar.gz}"
MODEL_DIR="./flue/pretrained_models/$MODEL_NAME"
MODEL_TAR="./flue/pretrained_models/$FILENAME"

# Check if the model folder or tar.gz file already exists
if [ -d "$MODEL_DIR" ] || [ -f "$MODEL_TAR" ]; then
    read -p "Model '$MODEL_NAME' already exists (folder or archive). Do you want to replace it? (y/n): " REPLACE
    if [[ "$REPLACE" != "y" && "$REPLACE" != "Y" ]]; then
        echo "Model already exists. Exiting."
        exit 0
    else
        rm -rf "$MODEL_DIR"
        rm -f "$MODEL_TAR"
        echo "Old model removed."
    fi
fi

# Download the model into pretrained_models folder
echo "Downloading $FILENAME..."
wget "$MODEL_URL" -O "$MODEL_TAR"
if [ $? -ne 0 ]; then
    echo "Download failed."
    exit 1
fi

# Extract the tar.gz file in pretrained_models folder
echo "Extracting $FILENAME..."
tar -xzf "$MODEL_TAR" -C ./flue/pretrained_models/

echo "Model downloaded and extracted to '$MODEL_DIR'."
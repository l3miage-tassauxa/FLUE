#!/bin/bash

# NB: This script is intended to be run in the 'FLUE' repository.

# Prompt user for the model download link
read -p "Enter the download link for the language model (.tar.gz or .zip): " MODEL_URL

# Extract the filename and model name (sanitize query string)
BASENAME=$(basename "$MODEL_URL")
CLEAN_NAME="${BASENAME%%\?*}"  # Remove query string
EXT="${CLEAN_NAME##*.}"
MODEL_NAME="${CLEAN_NAME%.tar.gz}"
if [[ "$CLEAN_NAME" == *.zip ]]; then
    MODEL_NAME="${CLEAN_NAME%.zip}"
fi
MODEL_DIR="./flue/pretrained_models/$MODEL_NAME"
MODEL_ARCHIVE="./flue/pretrained_models/$CLEAN_NAME"

# Check if the model folder or archive file already exists
if [ -d "$MODEL_DIR" ] || [ -f "$MODEL_ARCHIVE" ]; then
    read -p "Model '$MODEL_NAME' already exists (folder or archive). Do you want to replace it? (y/n): " REPLACE
    if [[ "$REPLACE" != "y" && "$REPLACE" != "Y" ]]; then
        echo "Model already exists. Exiting."
        exit 0
    else
        rm -rf "$MODEL_DIR"
        rm -f "$MODEL_ARCHIVE"
        echo "Old model removed."
    fi
fi

# Download the model into pretrained_models folder
echo "Downloading $CLEAN_NAME..."
wget "$MODEL_URL" -O "$MODEL_ARCHIVE"
if [ $? -ne 0 ]; then
    echo "Download failed."
    exit 1
fi

# Extract the archive in pretrained_models folder
echo "Extracting $CLEAN_NAME..."
if [[ "$CLEAN_NAME" == *.tar.gz ]]; then
    tar -xzf "$MODEL_ARCHIVE" -C ./flue/pretrained_models/
elif [[ "$CLEAN_NAME" == *.zip ]]; then
    unzip -q "$MODEL_ARCHIVE" -d ./flue/pretrained_models/
else
    echo "Unsupported file format: $CLEAN_NAME"
    exit 1
fi

echo "Model downloaded and extracted to '$MODEL_DIR'."
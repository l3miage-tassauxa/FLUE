#!/usr/bin/env bash
# inspired by prepare-data-cls.sh
# Copyright 2019 Hang Le
# hangtp.le@gmail.com

set -e

# Input parameters
DATA_DIR=$1
MODEL_DIR=$2
do_lower=$3

if [ $# -eq 3 ]
then
    echo "Running script ..."
else
    echo "3 arguments must be provided!"
    exit 1
fi

# Extraction
python flue/extract_split_cls.py --indir $DATA_DIR/raw/cls-acl10-unprocessed --outdir $DATA_DIR/processed --do_lower $do_lower

category="books dvd music"
splits="train valid test"

TOKENIZER=./tools/tokenize.sh
FASTBPE=./tools/fastBPE/fast
chmod +x $TOKENIZER
chmod +x $FASTBPE

CODES_PATH=$MODEL_DIR/codes
VOCAB_PATH=$MODEL_DIR/vocab

for cat in $category; do
    for split in $splits; do
        if [ ! -f $DATA_DIR/processed/$cat/${split}.tsv ]; then
            awk -F '\t' '{ print $1}' $DATA_DIR/processed/$cat/${split}_0.tsv \
            | $TOKENIZER 'fr' \
            > $DATA_DIR/processed/$cat/${split}.x

            awk -F '\t' '{ print $2}' $DATA_DIR/processed/$cat/${split}_0.tsv \
            > $DATA_DIR/processed/$cat/${split}.label

            $FASTBPE applybpe $DATA_DIR/processed/$cat/${split}.s1 $DATA_DIR/processed/$cat/${split}.x $CODES_PATH
            python preprocess.py $VOCAB_PATH $DATA_DIR/processed/$cat/$split.s1

            paste $DATA_DIR/processed/$cat/${split}.x $DATA_DIR/processed/$cat/${split}.label > $DATA_DIR/processed/$cat/${split}.tsv

            rm $DATA_DIR/processed/$cat/${split}_0.tsv $DATA_DIR/processed/$cat/${split}.x

            echo "Finished processing ${split} and saved to $DATA_DIR/processed/$cat."
        else
            echo 'Data has already been processed.'
        fi
    done
    echo 'Finished preparing data for category: '$cat
done

# CONCATENATION DES CATEGORIES
for split in $splits; do
    # Concatène tous les .s1, .label, .tsv
    cat $DATA_DIR/processed/books/${split}.s1 $DATA_DIR/processed/dvd/${split}.s1 $DATA_DIR/processed/music/${split}.s1 > $DATA_DIR/processed/${split}.s1
    cat $DATA_DIR/processed/books/${split}.label $DATA_DIR/processed/dvd/${split}.label $DATA_DIR/processed/music/${split}.label > $DATA_DIR/processed/${split}.labels
    cat $DATA_DIR/processed/books/${split}.tsv $DATA_DIR/processed/dvd/${split}.tsv $DATA_DIR/processed/music/${split}.tsv > $DATA_DIR/processed/${split}.tsv
done

# BINARISATION (création des .pth) avec preprocess.py
for split in $splits; do
    python preprocess.py $VOCAB_PATH $DATA_DIR/processed/${split}.s1
    python preprocess.py $VOCAB_PATH $DATA_DIR/processed/${split}.labels
done

# Création des liens symboliques ou copies pour les fichiers .label attendus
for split in $splits; do
    if [ -f $DATA_DIR/processed/${split}.labels ]; then
        cp $DATA_DIR/processed/${split}.labels $DATA_DIR/processed/${split}.label
    fi
done

echo "CLS data preparation and binarization complete."
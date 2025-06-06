#!/usr/bin/env bash
# Aurélien Tassaux

#Macros
DATA_DIR=./Data
MODEL_DIR=./Model
MODEL_PATH=$MODEL_DIR/model.pth

# Check isi le premier argument est fourni
if [ -z "$1" ]; then
        echo "Veuiller spécifier une tache."
        exit 1
fi
# Check si le deuxième argument est fourni
if [ -z "$2" ]; then
        echo "Veuiller spécifier si les librairies doivent être installées (true/false)."
        exit 1
fi

# Installe les libraries requises
if [ $2 == true ]; then
    # Check si on est dans le dossier flue
    if [ "$(basename "$PWD")" != "flue" ]; then
        echo "Veuillez positionner le terminal dans le dossier FLUE/flue."
        exit 1
    fi
    echo "Installing required libraries..."
    pip install -r ../requirements.txt
    cd ..
    cd tools
    git clone https://github.com/attardi/wikiextractor.git
    git clone https://github.com/moses-smt/mosesdecoder.git
    git clone https://github.com/glample/fastBPE.git
    cd fastBPE
    g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
    cd ..
    cd ..
    cd flue
    echo "Libraries installed."
else
    echo "Skipping library installation."
fi

# Lance les scripts de préparation des données et d'avaluation en fonction de la tâche spécifiée
case $1 in
    cls)
    echo "Getting CLS data..."
        ./get-data-cls.sh $DATA_DIR
        echo "Preparing CLS data..."
        ./flue/prepare-data-cls.sh $DATA_DIR $MODEL_PATH false
        ;;

    pawsx)
        echo "Getting PAWSX data..."
        ./get-data-xnli.sh $DATA_DIR
        echo "Preparing PAWSX data..."
        ./flue/prepare-data-pawsx.sh $DATA_DIR $MODEL_PATH false
        ;;

    xnli)
        echo "Getting XNLI data..."
        ./get-data-xnli.sh $DATA_DIR
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh $DATA_DIR $MODEL_PATH false
        echo "Running XNLI evaluation..."
        # pas vérifié a partir de ce moment là
        oarsub -I
        config='flue/examples/xnli_lr5e6_xlm_base_cased.cfg'
        source $config
        python flue/flue_xnli.py --exp_name $exp_name \
                        --exp_id $exp_id \
                        --dump_path $dump_path  \
                        --model_path $model_path  \
                        --data_path $data_path  \
                        --dropout $dropout \
                        --transfer_tasks $transfer_tasks \
                        --optimizer_e adam,lr=$lre \
                        --optimizer_p adam,lr=$lrp \
                        --finetune_layers $finetune_layer \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
        oarsub -c
        echo "XNLI evaluation completed."
        ;;
    vsd)
        echo "Getting VSD data..."
        FSE_DIR=./Data/FSE-1.1-191210
        python prepare_data.py --data $FSE_DIR --output $DATA_DIR
        echo "Preparing VSD data..."
        ./flue/prepare-data-vsd.sh $DATA_DIR $MODEL_PATH false
        ;;
    *)
        echo "Veuiller spécifier une tache valide."
        echo "Tâches valides: cls, pawsx, xnli"
        exit 1
        ;;
esac

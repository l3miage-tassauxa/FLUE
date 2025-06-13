#!/usr/bin/env bash
# Aurélien Tassaux

# Macros
DATA_DIR=./flue/data
MODEL_DIR=./flue/pretrained_models/
MODEL_PATH=$MODEL_DIR

# Check si le premier argument est fourni
if [ -z "$1" ]; then
        echo "Veuiller spécifier une tache."
        exit 1
fi
# Check si le deuxième argument est fourni
if [ -z "$2" ]; then
        echo "Veuiller spécifier si les librairies doivent être installées (true/false)."
        exit 1
fi

# Check si on est dans le dossier FLUE
if [ "$(basename "$PWD")" != "FLUE" ]; then
    echo "Veuillez positionner le terminal dans le dossier FLUE, racine du git."
    exit 1
fi

# Lance les scripts de préparation des données et d'évaluation en fonction de la tâche spécifiée
case $1 in
    cls)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./requirements.txt
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Getting CLS data..."
        ./flue/get-data-cls.sh $DATA_DIR
        echo "Preparing CLS data..."
        ./flue/prepare-data-cls.sh $DATA_DIR $MODEL_PATH false
        ;;

    pawsx)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./requirements.txt
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Getting PAWSX data..."
        ./flue/get-data-xnli.sh $DATA_DIR
        echo "Preparing PAWSX data..."
        ./flue/prepare-data-pawsx.sh $DATA_DIR $MODEL_PATH false
        ;;

    xnli-flaubert)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/xnli-requirements.txt
            cd ./tools
            git clone https://github.com/attardi/wikiextractor.git
            git clone https://github.com/moses-smt/mosesdecoder.git
            git clone https://github.com/glample/fastBPE.git
            cd ./fastBPE
            g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
            cd ../..
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/prepare-data-xnli.sh ./flue/flue_xnli.py
        echo "Getting XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_PATH false
        echo "Running XNLI evaluation..."
        config='flue/examples/xnli_lr5e6_xlm_base_cased.cfg'
        source $config
        python ./flue/flue_xnli.py --exp_name $exp_name \
                        --exp_id $exp_id \
                        --dump_path $dump_path  \
                        --model_path $model_path  \
                        --data_path $data_path  \
                        --dropout $dropout \
                        --transfer_tasks $transfer_tasks \
                        --optimizer_e adam,lr=$lre \
                        --optimizer_p adam,lr=$lrp \
                        --finetune_layers $finetune_layers \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
        ;;
    xnli-pentagruel)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/xnli-requirements.txt
            cd ./tools
            git clone https://github.com/attardi/wikiextractor.git
            git clone https://github.com/moses-smt/mosesdecoder.git
            git clone https://github.com/glample/fastBPE.git
            cd ./fastBPE
            g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
            cd ../..
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/prepare-data-xnli.sh ./flue/flue_xnli.py
        echo "Getting XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_PATH false
        echo "Running XNLI evaluation..."
        config='flue/examples/xnli_lr5e6_xlm_base_cased.cfg'
        source $config
        python ./flue/flue_xnli.py --exp_name $exp_name \
                        --exp_id $exp_id \
                        --dump_path $dump_path  \
                        --model_path $model_path  \
                        --data_path $data_path  \
                        --dropout $dropout \
                        --transfer_tasks $transfer_tasks \
                        --optimizer_e adam,lr=$lre \
                        --optimizer_p adam,lr=$lrp \
                        --finetune_layers $finetune_layers \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
        ;;
    parse)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/xnli-requirements.txt
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Getting Parse data..."
        ./flue/get-data-parse.sh $DATA_DIR
        echo "Preparing Parse data..."
        ./flue/prepare-data-parse.sh $DATA_DIR $MODEL_PATH false
        ;;
    wsd)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./requirements.txt
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Getting WSD data..."
        FSE_DIR=./Data/FSE-1.1-191210
        python ./flue/prepare_data.py --data $FSE_DIR --output $DATA_DIR
        echo "Preparing WSD data..."
        ./flue/prepare-data-wsd.sh $DATA_DIR $MODEL_PATH false
        ;;
    *)
        echo "Veuiller spécifier une tache valide."
        echo "Tâches valides: cls, pawsx, xnli-flaubert, parse, wsd"
        exit 1
        ;;
esac
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
    cls-XLM)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/XLM-requirements.txt
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
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Getting CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must make a demand for the data at https://zenodo.org/record/3251672"
            echo "and place the file in $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        else
            echo "Unzipping data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data unzipped."
        fi
        echo "Preparing CLS data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_small_cased_xlm true
        echo "Running CLS evaluation..."
        config='flue/examples/cls_books_lr5e6_xlm_base_cased.cfg'
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
                        --finetune_layers $finetune_layers \
                        --batch_size $batch_size \
                        --n_epochs $num_epochs \
                        --epoch_size $epoch_size \
                        --max_len $max_len \
                        --max_vocab $max_vocab
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

    xnli-XLM)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/XLM-requirements.txt
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
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Getting XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        # Use true for the third argument with Flaubert/XLM models 
        #(flaubert_base_cased_xlm, flaubert_small_cased_xlm, etc.)
        #Use false only if you want to use the default vocab/codes.
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_PATH true 
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
    xnli-HF)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/extract_xnli.py ./flue/flue_xnli.py
        echo "Getting XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_PATH true 
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
        echo "Calculating accuracy from task 3 predictions..."
        python ./flue/accuracy_from_task3.py
        echo "End of XNLI evaluation."
        ;;
    xnli-HF)
        if [ $2 == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Skipping library installation."
        fi
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/extract_xnli.py ./flue/binarize.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        chmod +x ./flue/accuracy_from_hf.py ./flue/examples/xnli_lr5e6_hf_base_uncased.cfg
        echo "Getting XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        python flue/extract_xnli.py --indir $DATA_DIR/xnli/processed \
                                 --outdir $DATA_DIR/xnli/processed \
                                 --do_lower false
        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/xnli/processed/
        echo "Running XNLI evaluation..."
        config='flue/examples/xnli_lr5e6_hf_base_uncased.cfg'
        source $config
        python tools/transformers/examples/pytorch/text-classification/run_glue.py \
                                        --train_file $data_dir/train.csv \
                                        --validation_file $data_dir/valid.csv \
                                        --test_file $data_dir/test.csv \
                                        --model_name_or_path $model_name_or_path \
                                        --output_dir $output_dir \
                                        --max_seq_length $max_seq_length \
                                        --do_train \
                                        --do_eval \
                                        --do_predict \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --save_steps $save_steps \
                                        --per_device_train_batch_size $batch_size \
                                        --per_device_eval_batch_size $batch_size \
                                        --weight_decay $weight_decay \
                                        --warmup_steps $warmup_steps \
                                        --gradient_accumulation_steps $gradient_accumulation_steps \
                                        --eval_strategy $eval_strategy \
                                        --save_strategy $save_strategy \
                                        --overwrite_output_dir \
                                        |& tee output.log
        echo "Calculating accuracy from Hugging Face predictions..."
        echo "Validation accuracy from training:"
            python -c "import json; data=json.load(open('$output_dir/eval_results.json')); print(f\"Validation accuracy: {data['eval_accuracy']*100:.2f}% on {data['eval_samples']} examples\")"
        echo "Test accuracy from predictions:"
            python flue/accuracy_from_hf.py --predictions_file $output_dir/predict_results_None.txt --labels_file $DATA_DIR/xnli/processed/test.label
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
        echo "Tâches valides: cls-XLM, pawsx, xnli-HF, xnli-XLM, parse, wsd"
        exit 1
        ;;
esac
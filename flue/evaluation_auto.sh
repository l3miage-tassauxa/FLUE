#!/usr/bin/env bash
# Aurélien Tassaux

# Macros
DATA_DIR=./flue/data
MODEL_DIR=./flue/pretrained_models/
MODEL_PATH=$MODEL_DIR

# Check first argument (task)
if [ -z "$1" ]; then
        echo "Usage: ./evaluation_auto.sh <task> <install_libs> <config_file>"
        echo "Tasks: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-XLM, pawsx-HF, parse, wsd"
        echo "Install libs: true/false"
        echo "Config file: path to a custom configuration file"
        exit 1
fi

# Parameters
TASK=$1
INSTALL_LIBS=$2
CUSTOM_CONFIG=$3 # Custom configuration file

echo "=== FLUE Evaluation ==="
echo "Task: $TASK"
echo "Library installation: $INSTALL_LIBS"
if [ ! -z "$CUSTOM_CONFIG" ]; then
    echo "Custom configuration: $CUSTOM_CONFIG"
fi

# Check current directory
if [ "$(basename "$PWD")" != "FLUE" ]; then
    echo "Please position the terminal in the FLUE directory, project root."
    exit 1
fi

# Launch according to task
case $TASK in
    cls-books-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
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
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You need to request access to the data at https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Data is already extracted."
        else
            echo "Extracting data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data extracted."
        fi
        echo "Preparing CLS books data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_base_cased_xlm_books true
        echo "Starting CLS books evaluation..."
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
        echo "Calculating accuracy from books task predictions..."
        python3 flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_books_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/cls/processed/books/test.label --format xlm --task cls
        ;;
    cls-music-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
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
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You need to request access to the data at https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Data is already extracted."
        else
            echo "Extracting data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data extracted."
        fi
        echo "Preparing CLS music data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_base_cased_xlm_music true
        echo "Starting CLS music evaluation..."
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
        echo "Calculating accuracy from music task predictions..."
        python3 flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_music_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/cls/processed/music/test.label --format xlm --task cls
        ;;
    cls-dvd-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
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
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You need to request access to the data at https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Data is already extracted."
        else
            echo "Extracting data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data extracted."
        fi
        echo "Preparing CLS dvd data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_base_cased_xlm_dvd true
        echo "Starting CLS DVD evaluation..."
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
        echo "Calculating accuracy from DVD task predictions..."
        python3 flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_dvd_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/cls/processed/dvd/test.label --format xlm --task cls
        ;;
    cls-XLM)
        echo "The cls-XLM task has been split into three distinct tasks:"
        echo "  - cls-books-XLM for books evaluation"
        echo "  - cls-music-XLM for music evaluation"  
        echo "  - cls-dvd-XLM for DVD evaluation"
        echo "Please use one of these specific tasks."
        exit 1
        ;;
    cls-books-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You need to request access to the data at https://zenodo.org/record/3251672"
            echo "et placer le fichier 'cls-acl10-unprocessed.tar' dans $DATA_DIR/cls/raw"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Data is already extracted."
        else
            echo "Extracting data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data extracted."
        fi
        echo "Preparing CLS books data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower false \
                                 --use_hugging_face true
        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/books/
        echo "Starting CLS books evaluation..."
        source $config
        python tools/transformers/examples/pytorch/text-classification/run_glue.py \
            --model_name_or_path $model_name_or_path \
            --output_dir $output_dir \
            --overwrite_output_dir \
            --max_seq_length $max_seq_length \
            --do_train \
            --do_eval \
            --learning_rate $lr \
            --num_train_epochs $epochs \
            --save_steps $save_steps \
            --fp16 \
            --train_file $train_file \
            --validation_file $validation_file \
            --test_file $test_file \
            --do_predict \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size $batch_size 

        echo "Calculating accuracy from Hugging Face results..."
        echo "Evaluation results with confidence interval:"
            python3 flue/accuracy_calculator.py --eval_results /home/getalp/tassauxa/FLUE/FLUE/flue/experiments/flaubert/cls_hf_flaubert_flaubert_base_cased/lr_5e-6/eval_results.json
        ;;
    cls-music-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You need to request access to the data at https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Data is already extracted."
        else
            echo "Extracting data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data extracted."
        fi
        echo "Preparing CLS music data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower false \
                                 --use_hugging_face true
        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/music/
        echo "Starting CLS music evaluation..."
        source $config
        python tools/transformers/examples/pytorch/text-classification/run_glue.py \
            --model_name_or_path $model_name_or_path \
            --output_dir $output_dir \
            --overwrite_output_dir \
            --max_seq_length $max_seq_length \
            --do_train \
            --do_eval \
            --learning_rate $lr \
            --num_train_epochs $epochs \
            --save_steps $save_steps \
            --fp16 \
            --train_file $train_file \
            --validation_file $validation_file \
            --test_file $test_file \
            --do_predict \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size $batch_size 

        echo "Calculating accuracy from Hugging Face results..."
        echo "Evaluation results with confidence interval:"
            python3 flue/accuracy_calculator.py --eval_results /home/getalp/tassauxa/FLUE/FLUE/flue/experiments/flaubert/cls_hf_flaubert_flaubert_base_cased/lr_5e-6/eval_results.json
        ;;
    cls-dvd-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You need to request access to the data at https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Data is already extracted."
        else
            echo "Extracting data..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Data extracted."
        fi
        echo "Preparing CLS dvd data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower false \
                                 --use_hugging_face true
        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/dvd/
        echo "Starting CLS dvd evaluation..."
        source $config
        python tools/transformers/examples/pytorch/text-classification/run_glue.py \
            --model_name_or_path $model_name_or_path \
            --output_dir $output_dir \
            --overwrite_output_dir \
            --max_seq_length $max_seq_length \
            --do_train \
            --do_eval \
            --learning_rate $lr \
            --num_train_epochs $epochs \
            --save_steps $save_steps \
            --fp16 \
            --train_file $train_file \
            --validation_file $validation_file \
            --test_file $test_file \
            --do_predict \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size $batch_size

        echo "Calculating accuracy from Hugging Face results..."
        echo "Evaluation results with confidence interval:"
            python3 flue/accuracy_calculator.py --eval_results /home/getalp/tassauxa/FLUE/FLUE/flue/experiments/flaubert/cls_hf_flaubert_flaubert_base_cased/lr_5e-6/eval_results.json
        ;;
    pawsx-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-pawsx.sh ./flue/flue_xnli.py ./flue/get-data-pawsx.sh
        chmod +x ./flue/accuracy_calculator.py
        
        echo "Retrieving PAWSX data..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Preparing PAWSX data..."
        ./flue/prepare-data-pawsx.sh $DATA_DIR $MODEL_PATH false

        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/pawsx/processed/
        echo "Starting PAWSX evaluation..."
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
        ;;
    pawsx-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/get-data-pawsx.sh ./flue/extract_pawsx.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Retrieving PAWSX data..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Preparing PAWSX data..."
        python flue/extract_pawsx.py --indir $DATA_DIR/pawsx/raw/x-final \
                             --outdir $DATA_DIR/pawsx/processed \
                             --use_hugging_face True

        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/pawsx/processed/
        echo "Starting PAWSX evaluation..."
        source $config
        python tools/transformers/examples/pytorch/text-classification/run_glue.py \
            --model_name_or_path $model_name_or_path \
            --output_dir $output_dir \
            --overwrite_output_dir \
            --max_seq_length $max_seq_length \
            --do_train \
            --do_eval \
            --learning_rate $lr \
            --num_train_epochs $epochs \
            --save_steps $save_steps \
            --fp16 \
            --train_file $train_file \
            --validation_file $validation_file \
            --test_file $test_file \
            --do_predict \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size $batch_size 
    ;;
    xnli-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
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
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/prepare-data-xnli.sh ./flue/flue_xnli.py ./flue/extract_xnli.py
        chmod +x ./flue/pretrained_models/flaubert_base_cased_xlm/*
        echo "Retrieving XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_PATH true 
        echo "Starting XNLI evaluation..."
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
        python3 flue/accuracy_calculator.py --predictions_file ./experiments/xnli_xlm_base_cased/dropout_0.1_lre_0.000005_lrp_0.000005/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/xnli/processed/test.label --format xlm --task xnli
        echo "XNLI evaluation completed."
        ;;
    xnli-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/extract_xnli.py 
        chmod +x ./flue/data/hg_data_tsv_to_csv.py ./flue/accuracy_calculator.py

        echo "Retrieving XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli

        echo "Preparing XNLI data..."
        python flue/extract_xnli.py --indir $DATA_DIR/xnli/processed \
                                 --outdir $DATA_DIR/xnli/processed \
                                 --do_lower false
        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/xnli/processed/
        
        echo "Starting XNLI evaluation..."
        source $config
        python tools/transformers/examples/pytorch/text-classification/run_glue.py \
            --model_name_or_path $model_name_or_path \
            --output_dir $output_dir \
            --overwrite_output_dir \
            --max_seq_length $max_seq_length \
            --do_train \
            --do_eval \
            --learning_rate $lr \
            --num_train_epochs $epochs \
            --save_steps $save_steps \
            --fp16 \
            --train_file $train_file \
            --validation_file $validation_file \
            --test_file $test_file \
            --do_predict \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size $batch_size 

        echo "Calculating accuracy from Hugging Face results..."
        echo "Evaluation results with confidence interval:"
            python3 flue/accuracy_calculator.py --eval_results /home/getalp/tassauxa/FLUE/FLUE/flue/experiments/flaubert/cls_hf_flaubert_flaubert_base_cased/lr_5e-6/eval_results.json
        ;;
    
    parse)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/xnli-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        echo "Retrieving Parse data..."
        ./flue/get-data-parse.sh $DATA_DIR
        echo "Preparing Parse data..."
        ./flue/prepare-data-parse.sh $DATA_DIR $MODEL_PATH false
        ;;
    wsd)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: configuration file '$config' does not exist in flue/examples directory."
                exit 1
            fi
        fi
        
        echo "Adding execution permissions to WSD scripts..."
        chmod +x ./flue/wsd/verbs/flue_vsd.py ./flue/wsd/verbs/run_model.py ./flue/wsd/verbs/prepare_data.py ./flue/wsd/verbs/wsd_evaluation.py
        
        echo "Checking WSD data..."
        if [ ! -d "$DATA_DIR/wsd/FSE-1.1-10_12_19" ]; then
            echo "Error: WSD data is not available in $DATA_DIR/wsd/"
            echo "Please download the FrenchSemEval (FSE) dataset from http://www.llf.cnrs.fr/dataset/fse/"
            echo "et l'extraire dans le dossier $DATA_DIR/wsd/"
            exit 1
        else
            echo "WSD data found."
        fi
        
        # WSD data preparation
        echo "Preparing WSD data for evaluation..."
        cd flue/wsd/verbs
        python prepare_data.py --data ../../../$DATA_DIR/wsd/FSE-1.1-10_12_19/FSE-1.1-191210 --output ../../../$DATA_DIR/wsd/processed
        cd ../../..
        
        # Launch WSD evaluation with specified model
        echo "Starting WSD evaluation..."
        cd flue/wsd/verbs
        
        # Automatic device detection (GPU if available, otherwise CPU)
        DEVICE=-1
        if command -v nvidia-smi >/dev/null 2>&1; then
            if nvidia-smi >/dev/null 2>&1; then
                DEVICE=0
                echo "GPU detected, using GPU."
            else
                echo "GPU not available, using CPU."
            fi
        else
            echo "NVIDIA not detected, using CPU."
        fi
        
        # python flue_vsd.py --exp_name wsd_${MODEL_NAME}_evaluation \
        #                   --model $MODEL_NAME \
        #                   --data ../../../$DATA_DIR/wsd/processed \
        #                   --padding 80 \
        #                   --batchsize 32 \
        #                   --device $DEVICE \
        #                   --output ../../../flue/experiments/wsd_${MODEL_NAME} \
        #                   --output_logs ../../../flue/experiments/wsd_${MODEL_NAME}/evaluation_logs.csv \
        #                   --output_pred ../../../flue/experiments/wsd_${MODEL_NAME}/predictions.txt \
        #                   --output_score ../../../flue/experiments/wsd_${MODEL_NAME}/scores.csv
        # cd ../../..
        
        # echo "WSD evaluation completed."
        # echo "Results available in: ./flue/experiments/wsd_${MODEL_NAME}/"
        # if [ -f "./flue/experiments/wsd_${MODEL_NAME}/scores.csv" ]; then
        #     echo "Evaluation scores:"
        #     cat ./flue/experiments/wsd_${MODEL_NAME}/scores.csv
        # fi
        ;;
    *)
        echo "Please specify a valid task."
        echo "Tasks valides: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-Mlflow, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-HF, parse, wsd"
        exit 1
esac
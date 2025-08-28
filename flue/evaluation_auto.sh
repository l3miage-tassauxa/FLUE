#!/usr/bin/env bash
# Aurélien Tassaux & Manyl Tidjani

# Macros
DATA_DIR=./flue/data
MODEL_DIR=./flue/pretrained_models/

# Check the first argument (task)
if [ -z "$1" ]; then
        echo "Usage: ./evaluation_auto.sh <task> <install_libs> <config_file>"
        echo "Tasks: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-XLM, pawsx-HF, parse, wsd, mlflow-cls-books-HF"
        echo "Install libs: true/false"
        echo "Config file: path to a custom configuration file"
        exit 1
fi

# Parameters
# Parameters
TASK=$1
INSTALL_LIBS=$2
CUSTOM_CONFIG=$3

echo "=== FLUE Evaluation ==="
echo "Task: $TASK"
echo "Install libraries: $INSTALL_LIBS"
if [ ! -z "$CUSTOM_CONFIG" ]; then
    echo "Custom configuration: $CUSTOM_CONFIG"
    echo "Custom configuration: $CUSTOM_CONFIG"
fi

# Check the current directory
if [ "$(basename "$PWD")" != "FLUE" ]; then
    echo "Please position the terminal in the FLUE directory, the root of the project."
    exit 1
fi

# Launch based on the task
case $TASK in
    cls-books-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
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
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py
        chmod +x ./flue/pretrained_models/$model_name/*

        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Preparing CLS books data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_DIR/$model_name $do_lower

        echo "Launching CLS books evaluation..."
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

        echo "Calculating accuracy from CLS books predictions..."
        python3 flue/accuracy_calculator.py --predictions_file $output_dir/test.pred.$((num_epochs - 1)) --labels_file $DATA_DIR/cls/processed/books/test.label --format xlm --task cls
    ;;
    cls-music-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
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
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py
        chmod +x ./flue/pretrained_models/$model_name/*

        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi
        
        echo "Preparing CLS music data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_DIR/$model_name $do_lower

        echo "Launching CLS music evaluation..."
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

        echo "Calculating accuracy from CLS music predictions..."
        python3 flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_music_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file $DATA_DIR/cls/processed/music/test.label --format xlm --task cls
        ;;
    cls-dvd-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
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
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py
        chmod +x ./flue/pretrained_models/$model_name/*
        
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi
        echo "Preparing CLS dvd data..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_DIR/$model_name $do_lower

        echo "Launching CLS DVD evaluation..."
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

        echo "Calculating accuracy from CLS DVD predictions..."
        python3 flue/accuracy_calculator.py --predictions_file $output_dir/test.pred.$((num_epochs - 1)) --labels_file $DATA_DIR/cls/processed/dvd/test.label --format xlm --task cls
    ;;
    cls-books-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Preparing CLS books data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower $do_lower \
                                 --use_hugging_face true

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/books/

        echo "Launching CLS books evaluation..."
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
            python3 flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json
    ;;
    mlflow-cls-books-HF)
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
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        
        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Preparing CLS books data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower $do_lower \
                                 --use_hugging_face true

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/books/

        echo "Mlflow tracking: Launching CLS books evaluation..."
        cmd="python tools/transformers/examples/pytorch/text-classification/run_glue.py \
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
            --per_device_eval_batch_size $batch_size"

        python tools/mlflow/mlflow_finetuning.py --command "$cmd" --experiment "Text Classification (CLS) - Books" --model ${model_name} --tracking_uri ${mlflow_tracking_uri}
    ;;
    cls-music-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Retrieving CLS data..."
        
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Preparing CLS music data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower $do_lower \
                                 --use_hugging_face true

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/music/

        echo "Launching CLS music evaluation..."
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
            python3 flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json
        ;;
    mlflow-cls-music-HF)
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
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Mlflow tracking: Launching CLS music evaluation..."
        cmd="python tools/transformers/examples/pytorch/text-classification/run_glue.py \
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
            --per_device_eval_batch_size $batch_size"

        python tools/mlflow/mlflow_finetuning.py --command "$cmd" --experiment "Text Classification (CLS) - Music" --model ${model_name} --tracking_uri ${mlflow_tracking_uri}
    ;;
    cls-dvd-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        echo "Using configuration: $config"
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in the folder $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Preparing CLS dvd data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower $do_lower \
                                 --use_hugging_face true

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/dvd/
        
        echo "Launching CLS dvd evaluation..."
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
            python3 flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json
        ;;
    mlflow-cls-dvd-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Retrieving CLS data..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "You must request the data at https://zenodo.org/record/3251672"
            echo "and place the file 'cls-acl10-unprocessed.tar.gz' in $DATA_DIR/cls/raw/"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "The data is already decompressed."
        else
            echo "Decompressing data..."
            tar -xvf $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz -C $DATA_DIR/cls/raw/
            echo "Data decompressed."
        fi

        echo "Preparing CLS dvd data..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower $do_lower \
                                 --use_hugging_face true

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/dvd/

        echo "Mlflow tracking: Launching CLS dvd evaluation..."
        cmd="python tools/transformers/examples/pytorch/text-classification/run_glue.py \
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
            --per_device_eval_batch_size $batch_size"

        python tools/mlflow/mlflow_finetuning.py --command "$cmd" --experiment "Text Classification (CLS) - DVD" --model ${model_name} --tracking_uri ${mlflow_tracking_uri}
    ;;
    pawsx-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            pip install -r ./libraries/XLM-requirements.txt
            echo "Libraries installed."
        else
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/prepare-data-pawsx.sh ./flue/flue_xnli.py ./flue/get-data-pawsx.sh
        
        echo "Retrieving PAWSX data..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Preparing PAWSX data..."
        ./flue/prepare-data-pawsx.sh $DATA_DIR/pawsx $MODEL_DIR/$model_name $do_lower

        echo "Launching PAWSX evaluation..."
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
    pawsx-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
            echo "Installing required libraries..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Libraries installed."
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config
        
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/get-data-pawsx.sh ./flue/extract_pawsx.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Retrieving PAWSX data..."
        echo "Retrieving PAWSX data..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Preparing PAWSX data..."
        echo "Preparing PAWSX data..."
        python flue/extract_pawsx.py --indir $DATA_DIR/pawsx/raw/x-final \
                             --outdir $DATA_DIR/pawsx/processed \
                             --use_hugging_face True

        echo "Converting TSV files to CSV format..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/pawsx/processed/

        echo "Launching PAWSX evaluation..."
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
    mlflow-pawsx-HF)
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
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config

        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/get-data-pawsx.sh ./flue/extract_pawsx.py $DATA_DIR/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Retrieving PAWSX data..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Preparing PAWSX data..."
        python flue/extract_pawsx.py --indir $DATA_DIR/pawsx/raw/x-final \
                             --outdir $DATA_DIR/pawsx/processed \
                             --use_hugging_face True

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/pawsx/processed/

        echo "Mlflow tracking: Launching paraphrasing PAWSX evaluation..."
        cmd="python tools/transformers/examples/pytorch/text-classification/run_glue.py \
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
            --per_device_eval_batch_size $batch_size"

        python tools/mlflow/mlflow_finetuning.py --command "$cmd" --experiment "Paraphrasing - PAWSX" --model ${model_name} --tracking_uri ${mlflow_tracking_uri}
    ;;
    xnli-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Please specify whether libraries should be installed (true/false)."
            echo "Please specify whether libraries should be installed (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installing required libraries..."
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
            echo "Libraries installed."
        else
            echo "Library installation skipped."
            echo "Library installation skipped."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Error: The configuration file '$config' does not exist in the flue/examples directory."
                exit 1
            fi
        fi
        echo "Using configuration: $config"
        source $config
        
        echo "Adding execution permissions to scripts..."
        echo "Adding execution permissions to scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/prepare-data-xnli.sh ./flue/flue_xnli.py ./flue/extract_xnli.py
        chmod +x ./flue/pretrained_models/$model_name/*

        echo "Retrieving XNLI data..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_DIR/$model_name $do_lower 

        echo "Launching XNLI evaluation..."
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

        echo "Calculating accuracy from predictions.."
        python3 flue/accuracy_calculator.py --predictions_file $output_dir/test.pred.$((num_epochs - 1)) --labels_file $DATA_DIR/xnli/processed/test.label --format xlm --task xnli
        echo "End of XNLI evaluation."
        ;;
    xnli-HF)
        echo "Soon..."
        exit 1
        ;;
    
    parsing)
        echo "Soon..."
        exit 1
        ;;
    wsd)
        echo "Soon..."
        exit 1
        ;;
    *)
        echo "Please specify a valid task."
        echo "Valid tasks: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-HF, parsing, wsd"
        echo "Valid tasks with Mlflow: mlflow-cls-books-HF, mlflow-cls-music-HF, mlflow-cls-dvd-HF, mlflow-pawsx-HF"
        exit 1
esac
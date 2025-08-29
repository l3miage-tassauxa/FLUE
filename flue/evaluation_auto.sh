#!/usr/bin/env bash
# Aurélien Tassaux & Manyl Tidjani

# Macros
DATA_DIR=./flue/data
MODEL_DIR=./flue/pretrained_models/
EXAMPLES_DIR="flue/examples"

show_usage() {
    cat << EOF
    Usage: $0 <task> <install_libs> <install_lib> <config_file>

    Tasks:
        XLM based:
            - cls-books-XLM, cls-music-XLM, cls-dvd-XLM
            - xnli-XLM, pawsx-XLM
        
        Hugging Face based:
            - cls-books-HF, cls-music-HF, cls-dvd-HF
            - pawsx-HF, xnli-HF
        
        MLflow enabled:
            - mlflow-cls-books-HF, mlflow-cls-music-HF, mlflow-cls-dvd-HF
            - mlflow-pawsx-HF
    
        Others:
            - parsing, wsd

    Install libs: true/false
    Config file: path to configuration file in '$EXAMPLES_DIR/' directory

    Examples:
    $0 cls-books-XLM true cls_books_lr5e6_xlm_base_cased.cfg
    $0 cls-books-HF false cls_books_lr5e6_hf.cfg
EOF
}

if [[ $# -lt 3 ]]; then
    echo "Insufficient arguments provided"
    show_usage
    exit 1
fi

if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    show_usage
    exit 0
fi

# Parameters
TASK=$1
INSTALL_LIBS=$2
CUSTOM_CONFIG=$3

if [[ "$(basename "$PWD")" != "FLUE" ]]; then
    echo "Please run this script from the FLUE root directory"
    exit 1
fi

# Check if config file exists
if [[ -n "$CUSTOM_CONFIG" ]]; then
    config_path="$EXAMPLES_DIR/$CUSTOM_CONFIG"
    if [[ ! -f "$config_path" ]]; then
        echo "Configuration file '$config_path' not found"
        exit 1
    fi
    echo "Using configuration: $config_path"
    source $config_path
fi

# Dependency installation functions
install_xlm_dependencies() {
    echo "Installing XLM dependencies..."

    if ! pip install -r ./libraries/xlm-requirements.txt; then
        echo "Failed to install XLM requirements"
        exit 1
    fi
    
    cd "./tools"
    
    # Clone repositories if they don't exist
    local repos=(
        "https://github.com/attardi/wikiextractor.git"
        "https://github.com/moses-smt/mosesdecoder.git"
        "https://github.com/glample/fastBPE.git"
    )
    
    for repo in "${repos[@]}"; do
        local repo_name
        repo_name=$(basename "$repo" .git)
        if [[ ! -d "$repo_name" ]]; then
            echo "Cloning $repo_name..."
            if ! git clone "$repo"; then
                echo "Failed to clone $repo"
                exit 1
            fi
        else
            echo "$repo_name already exists, skipping clone"
        fi
    done
    
    # Build fastBPE
    cd "./fastBPE"    
    if ! g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast; then
        echo "Failed to build fastBPE"
        exit 1
    fi
    
    cd ../../
    
    echo "XLM dependencies installed successfully"
}

install_hf_dependencies() {
    echo "Installing Hugging Face dependencies..."
    
    if ! pip install -r ./libraries/hg-requirements.txt; then
        echo "Failed to install HF requirements"
        exit 1
    fi
    
    echo "Hugging Face dependencies installed successfully"
}

# Install dependencies based on task type
if [[ $INSTALL_LIBS == true ]]; then
    case "$TASK" in
        *-XLM)
            install_xlm_dependencies
            ;;
        *-HF|mlflow-*-HF)
            install_hf_dependencies
            ;;
    esac
else
    echo "Library installation skipped"
fi

# Permissions
echo "Adding execution permissions to scripts..."
chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py $DATA_DIR/hg_data_tsv_to_csv.py
chmod +x ./flue/prepare-data-pawsx.sh ./flue/get-data-pawsx.sh
chmod +x ./flue/get-data-xnli.sh ./flue/prepare-data-xnli.sh ./flue/flue_xnli.py ./flue/extract_xnli.py
chmod +x ./flue/accuracy_calculator.py

if [[ "$TASK" == *-XLM ]]; then
    chmod +r "./flue/pretrained_models/$model_name"/*
fi

# Launch based on the task
case $TASK in
    cls-books-XLM)
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
        python3 flue/accuracy_calculator.py --predictions_file $output_dir/test.pred.$((num_epochs - 1)) --labels_file $DATA_DIR/cls/processed/music/test.label --format xlm --task cls
        ;;
    cls-dvd-XLM)
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

        echo "Converting TSV files to CSV format..."
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

        echo "Converting TSV files to CSV format..."
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

        echo "Converting TSV files to CSV format..."
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

        echo "Converting TSV files to CSV format..."
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
        echo "Retrieving PAWSX data..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
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
        echo "task not yet implemented..."
        exit 1
        ;;
    
    parsing)
        echo "task not yet implemented..."
        exit 1
        ;;
    wsd)
        echo "task not yet implemented..."
        exit 1
        ;;
    *)
        echo "Please specify a valid task."
        echo "Valid tasks: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-HF, parsing, wsd"
        echo "Valid tasks with Mlflow: mlflow-cls-books-HF, mlflow-cls-music-HF, mlflow-cls-dvd-HF, mlflow-pawsx-HF"
        exit 1
esac
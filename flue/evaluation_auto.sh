#!/usr/bin/env bash
# Aurélien Tassaux

# Macros
DATA_DIR=./flue/data
MODEL_DIR=./flue/pretrained_models/
MODEL_PATH=$MODEL_DIR

# Vérification du premier argument (tâche)
if [ -z "$1" ]; then
        echo "Usage: ./evaluation_auto.sh <tâche> <installer_libs> <fichier_config>"
        echo "Tâches: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-XLM, pawsx-HF, parse, wsd"
        echo "Installer libs: true/false"
        echo "Fichier config: chemin vers un fichier de configuration personnalisé"
        exit 1
fi

# Paramètres
TASK=$1
INSTALL_LIBS=$2
CUSTOM_CONFIG=$3 # Fichier de configuration personnalisé

echo "=== Évaluation FLUE ==="
echo "Tâche: $TASK"
echo "Installation des librairies: $INSTALL_LIBS"
if [ ! -z "$CUSTOM_CONFIG" ]; then
    echo "Configuration personnalisée: $CUSTOM_CONFIG"
fi

# Vérification du dossier courant
if [ "$(basename "$PWD")" != "FLUE" ]; then
    echo "Veuillez positionner le terminal dans le dossier FLUE, racine du projet."
    exit 1
fi

# Lancement selon la tâche
case $TASK in
    cls-books-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/XLM-requirements.txt
            cd ./tools
            git clone https://github.com/attardi/wikiextractor.git
            git clone https://github.com/moses-smt/mosesdecoder.git
            git clone https://github.com/glample/fastBPE.git
            cd ./fastBPE
            g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
            cd ../..
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Les données sont déjà décompressées."
        else
            echo "Décompression des données..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Données décompressées."
        fi
        echo "Préparation des données CLS books..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_base_cased_xlm_books true
        echo "Lancement de l'évaluation CLS books..."
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
        echo "Calcul de la précision à partir des prédictions de la tâche books..."
        python flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_books_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/cls/processed/books/test.label --format xlm --task cls
        ;;
    cls-music-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/XLM-requirements.txt
            cd ./tools
            git clone https://github.com/attardi/wikiextractor.git
            git clone https://github.com/moses-smt/mosesdecoder.git
            git clone https://github.com/glample/fastBPE.git
            cd ./fastBPE
            g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
            cd ../..
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Les données sont déjà décompressées."
        else
            echo "Décompression des données..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Données décompressées."
        fi
        echo "Préparation des données CLS music..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_base_cased_xlm_music true
        echo "Lancement de l'évaluation CLS music..."
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
        echo "Calcul de la précision à partir des prédictions de la tâche music..."
        python flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_music_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/cls/processed/music/test.label --format xlm --task cls
        ;;
    cls-dvd-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/XLM-requirements.txt
            cd ./tools
            git clone https://github.com/attardi/wikiextractor.git
            git clone https://github.com/moses-smt/mosesdecoder.git
            git clone https://github.com/glample/fastBPE.git
            cd ./fastBPE
            g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
            cd ../..
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py
        chmod +x ./flue/pretrained_models/flaubert_small_cased_xlm/*
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Les données sont déjà décompressées."
        else
            echo "Décompression des données..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Données décompressées."
        fi
        echo "Préparation des données CLS dvd..."
        ./flue/prepare-data-cls.sh $DATA_DIR/cls $MODEL_PATH/flaubert_base_cased_xlm_dvd true
        echo "Lancement de l'évaluation CLS DVD..."
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
        echo "Calcul de la précision à partir des prédictions de la tâche DVD..."
        python flue/accuracy_calculator.py --predictions_file ./flue/experiments/cls_dvd_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/cls/processed/dvd/test.label --format xlm --task cls
        ;;
    cls-XLM)
        echo "La tâche cls-XLM a été séparée en trois tâches distinctes:"
        echo "  - cls-books-XLM pour l'évaluation sur les livres"
        echo "  - cls-music-XLM pour l'évaluation sur la musique"  
        echo "  - cls-dvd-XLM pour l'évaluation sur les DVD"
        echo "Veuillez utiliser une de ces tâches spécifiques."
        exit 1
        ;;
    cls-books-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier 'cls-acl10-unprocessed.tar' dans $DATA_DIR/cls/raw"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Les données sont déjà décompressées."
        else
            echo "Décompression des données..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Données décompressées."
        fi
        echo "Préparation des données CLS books..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower false \
                                 --use_hugging_face true
        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/books/
        
        echo "Configuration de MLflow Enhanced (maximum data capture)..."
        export MLFLOW_TRACKING_URI="file://$(pwd)/flue/mlruns"
        export MLFLOW_EXPERIMENT_NAME="FLUE_CLS_Books_HF"
        
        echo "Lancement de l'évaluation CLS books avec MLflow Enhanced..."
        source $config
        python flue/enhanced_mlflow_run_glue.py \
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

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "Erreur: L'entraînement a échoué. Arrêt du script."
            exit 1
        fi

        echo "Calcul de la précision à partir des résultats Hugging Face..."
        echo "Résultats d'évaluation avec intervalle de confiance:"
        accuracy_output=$(python flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json)
        echo "$accuracy_output"
        
        echo "Logging des résultats dans MLflow..."
        python flue/log_to_mlflow.py "$output_dir/eval_results.json" "cls_books_hf" "$model_name_or_path" "$lr" "$epochs" "$batch_size" "$accuracy_output"
        ;;
    cls-music-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Les données sont déjà décompressées."
        else
            echo "Décompression des données..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Données décompressées."
        fi
        echo "Préparation des données CLS music..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower false \
                                 --use_hugging_face true
        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/music/
        
        echo "Configuration de MLflow..."
        export MLFLOW_TRACKING_URI="file://$(pwd)/flue/mlruns"
        export MLFLOW_EXPERIMENT_NAME="FLUE_CLS_Music"
        
        echo "Lancement de l'évaluation CLS music avec MLflow Enhanced..."
        source $config
        python flue/enhanced_mlflow_run_glue.py \
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

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "Erreur: L'entraînement a échoué. Arrêt du script."
            exit 1
        fi

        echo "Calcul de la précision à partir des résultats Hugging Face..."
        echo "Résultats d'évaluation avec intervalle de confiance:"
        accuracy_output=$(python flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json)
        echo "$accuracy_output"
        
        echo "Logging des résultats dans MLflow..."
        python flue/log_to_mlflow.py "$output_dir/eval_results.json" "cls_music" "$model_name_or_path" "$lr" "$epochs" "$batch_size" "$accuracy_output"
        ;;
    cls-dvd-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
        elif [ -d "$DATA_DIR/cls/raw/cls-acl10-unprocessed" ]; then
            echo "Les données sont déjà décompressées."
        else
            echo "Décompression des données..."
            tar -xvf ./flue/data/cls/raw/cls-acl10-unprocessed.tar.gz -C ./flue/data/cls/raw/
            echo "Données décompressées."
        fi
        echo "Préparation des données CLS dvd..."
        python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                                 --outdir $DATA_DIR/cls/processed \
                                 --do_lower false \
                                 --use_hugging_face true
        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/dvd/
        
        echo "Configuration de MLflow..."
        export MLFLOW_TRACKING_URI="file://$(pwd)/flue/mlruns"
        export MLFLOW_EXPERIMENT_NAME="FLUE_CLS_DVD"
        
        echo "Lancement de l'évaluation CLS dvd avec MLflow Enhanced..."
        source $config
        python flue/enhanced_mlflow_run_glue.py \
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

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "Erreur: L'entraînement a échoué. Arrêt du script."
            exit 1
        fi

        echo "Calcul de la précision à partir des résultats Hugging Face..."
        echo "Résultats d'évaluation avec intervalle de confiance:"
        accuracy_output=$(python flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json)
        echo "$accuracy_output"
        
        echo "Logging des résultats dans MLflow..."
        python flue/log_to_mlflow.py "$output_dir/eval_results.json" "cls_dvd" "$model_name_or_path" "$lr" "$epochs" "$batch_size" "$accuracy_output"
        ;;
    pawsx-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"

        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-pawsx.sh ./flue/flue_xnli.py ./flue/get-data-pawsx.sh
        chmod +x ./flue/accuracy_calculator.py
        
        echo "Récupération des données PAWSX..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Préparation des données PAWSX..."
        ./flue/prepare-data-pawsx.sh $DATA_DIR $MODEL_PATH false

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/pawsx/processed/
        echo "Lancement de l'évaluation PAWSX..."
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
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/get-data-pawsx.sh ./flue/extract_pawsx.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_calculator.py

        echo "Récupération des données PAWSX..."
        ./flue/get-data-pawsx.sh $DATA_DIR/pawsx
        echo "Préparation des données PAWSX..."
        python flue/extract_pawsx.py --indir $DATA_DIR/pawsx/raw/x-final \
                             --outdir $DATA_DIR/pawsx/processed \
                             --use_hugging_face True

        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/pawsx/processed/
        
        echo "Configuration de MLflow..."
        export MLFLOW_TRACKING_URI="file://$(pwd)/flue/mlruns"
        export MLFLOW_EXPERIMENT_NAME="FLUE_PAWSX"
        
        echo "Lancement de l'évaluation PAWSX avec MLflow Enhanced..."
        source $config
        python flue/enhanced_mlflow_run_glue.py \
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

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "Erreur: L'entraînement a échoué. Arrêt du script."
            exit 1
        fi

        echo "Calcul de la précision à partir des résultats Hugging Face..."
        echo "Résultats d'évaluation avec intervalle de confiance:"
        accuracy_output=$(python flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json)
        echo "$accuracy_output"
        
        echo "Logging des résultats dans MLflow..."
        python flue/log_to_mlflow.py "$output_dir/eval_results.json" "pawsx" "$model_name_or_path" "$lr" "$epochs" "$batch_size" "$accuracy_output" 
    ;;
    xnli-XLM)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/XLM-requirements.txt
            cd ./tools
            git clone https://github.com/attardi/wikiextractor.git
            git clone https://github.com/moses-smt/mosesdecoder.git
            git clone https://github.com/glample/fastBPE.git
            cd ./fastBPE
            g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
            cd ../..
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/prepare-data-xnli.sh ./flue/flue_xnli.py ./flue/extract_xnli.py
        chmod +x ./flue/pretrained_models/flaubert_base_cased_xlm/*
        echo "Récupération des données XNLI..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli
        echo "Préparation des données XNLI..."
        ./flue/prepare-data-xnli.sh $DATA_DIR/xnli $MODEL_PATH true 
        echo "Lancement de l'évaluation XNLI..."
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
        echo "Calcul de la précision à partir des prédictions de la tâche 3..."
        python flue/accuracy_calculator.py --predictions_file ./experiments/xnli_xlm_base_cased/dropout_0.1_lre_0.000005_lrp_0.000005/test.pred.$((num_epochs - 1)) --labels_file ./flue/data/xnli/processed/test.label --format xlm --task xnli
        echo "Fin de l'évaluation XNLI."
        ;;
    xnli-HF)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/get-data-xnli.sh ./flue/extract_xnli.py 
        chmod +x ./flue/data/hg_data_tsv_to_csv.py ./flue/accuracy_calculator.py

        echo "Récupération des données XNLI..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli

        echo "Préparation des données XNLI..."
        python flue/extract_xnli.py --indir $DATA_DIR/xnli/processed \
                                 --outdir $DATA_DIR/xnli/processed \
                                 --do_lower false
        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/xnli/processed/
        
        echo "Configuration de MLflow..."
        export MLFLOW_TRACKING_URI="file://$(pwd)/flue/mlruns"
        export MLFLOW_EXPERIMENT_NAME="FLUE_XNLI"
        
        echo "Lancement de l'évaluation XNLI avec MLflow Enhanced..."
        source $config
        python flue/enhanced_mlflow_run_glue.py \
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

        # Check if training was successful
        if [ $? -ne 0 ]; then
            echo "Erreur: L'entraînement a échoué. Arrêt du script."
            exit 1
        fi

        echo "Calcul de la précision à partir des résultats Hugging Face..."
        echo "Résultats d'évaluation avec intervalle de confiance:"
        accuracy_output=$(python flue/accuracy_calculator.py --eval_results $output_dir/eval_results.json)
        echo "$accuracy_output"
        
        echo "Logging des résultats dans MLflow..."
        python flue/log_to_mlflow.py "$output_dir/eval_results.json" "xnli" "$model_name_or_path" "$lr" "$epochs" "$batch_size" "$accuracy_output"
        ;;
    
    parse)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/xnli-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        echo "Récupération des données Parse..."
        ./flue/get-data-parse.sh $DATA_DIR
        echo "Préparation des données Parse..."
        ./flue/prepare-data-parse.sh $DATA_DIR $MODEL_PATH false
        ;;
    wsd)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./libraries/hg-requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
            if [ ! -f "$config" ]; then
                echo "Erreur : le fichier de configuration '$config' n'existe pas dans le répertoire flue/examples."
                exit 1
            fi
        fi
        
        echo "Ajout des droits d'exécution aux scripts WSD..."
        chmod +x ./flue/wsd/verbs/flue_vsd.py ./flue/wsd/verbs/run_model.py ./flue/wsd/verbs/prepare_data.py ./flue/wsd/verbs/wsd_evaluation.py
        
        echo "Vérification des données WSD..."
        if [ ! -d "$DATA_DIR/wsd/FSE-1.1-10_12_19" ]; then
            echo "Erreur: Les données WSD ne sont pas disponibles dans $DATA_DIR/wsd/"
            echo "Veuillez télécharger le dataset FrenchSemEval (FSE) depuis http://www.llf.cnrs.fr/dataset/fse/"
            echo "et l'extraire dans le dossier $DATA_DIR/wsd/"
            exit 1
        else
            echo "Données WSD trouvées."
        fi
        
        # Préparation des données WSD
        echo "Préparation des données WSD pour l'évaluation..."
        cd flue/wsd/verbs
        python prepare_data.py --data ../../../$DATA_DIR/wsd/FSE-1.1-10_12_19/FSE-1.1-191210 --output ../../../$DATA_DIR/wsd/processed
        cd ../../..
        
        # Lancement de l'évaluation WSD avec le modèle spécifié
        echo "Lancement de l'évaluation WSD..."
        cd flue/wsd/verbs
        
        # Détection automatique du device (GPU si disponible, sinon CPU)
        DEVICE=-1
        if command -v nvidia-smi >/dev/null 2>&1; then
            if nvidia-smi >/dev/null 2>&1; then
                DEVICE=0
                echo "GPU détecté, utilisation du GPU."
            else
                echo "GPU non disponible, utilisation du CPU."
            fi
        else
            echo "NVIDIA non détecté, utilisation du CPU."
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
        
        # echo "Évaluation WSD terminée."
        # echo "Résultats disponibles dans: ./flue/experiments/wsd_${MODEL_NAME}/"
        # if [ -f "./flue/experiments/wsd_${MODEL_NAME}/scores.csv" ]; then
        #     echo "Scores d'évaluation:"
        #     cat ./flue/experiments/wsd_${MODEL_NAME}/scores.csv
        # fi
        ;;
    *)
        echo "Veuiller spécifier une tache valide."
        echo "Tâches valides: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-books-HF, cls-music-HF, cls-dvd-HF, xnli-HF, xnli-XLM, pawsx-HF, parse, wsd"
        exit 1
esac
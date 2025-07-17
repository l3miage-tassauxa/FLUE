#!/usr/bin/env bash
# Aurélien Tassaux

# Macros
DATA_DIR=./flue/data
MODEL_DIR=./flue/pretrained_models/
MODEL_PATH=$MODEL_DIR

# Vérification du premier argument (tâche)
if [ -z "$1" ]; then
        echo "Usage: ./evaluation_auto.sh <tâche> <installer_libs> [nom_modèle] [fichier_config]"
        echo "Tâches: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-HF, xnli-HF, xnli-XLM, pawsx, parse, wsd"
        echo "Installer libs: true/false"
        echo "Nom du modèle: flaubert_base_cased, flaubert_base_uncased, camembert_base, etc."
        echo "Fichier config: chemin vers un fichier de configuration personnalisé (optionnel)"
        exit 1
fi

# Paramètres
TASK=$1
INSTALL_LIBS=$2
MODEL_NAME=${3:-"flaubert_base_cased"}  # Modèle par défaut
CUSTOM_CONFIG=$4  # Fichier de configuration personnalisé (optionnel)

echo "=== Évaluation FLUE ==="
echo "Tâche: $TASK"
echo "Modèle: $MODEL_NAME"
echo "Installation des librairies: $INSTALL_LIBS"
if [ ! -z "$CUSTOM_CONFIG" ]; then
    echo "Configuration personnalisée: $CUSTOM_CONFIG"
fi

# Vérification du dossier courant
if [ "$(basename "$PWD")" != "FLUE" ]; then
    echo "Veuillez positionner le terminal dans le dossier FLUE, racine du git."
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
        else
            config="flue/examples/cls_books_lr5e6_xlm_base_cased.cfg"
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
        python flue/accuracy_from_task1.py --logits_file ./flue/experiments/cls_books_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.29 --labels_file ./flue/data/cls/processed/books/test.label        ;;
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
        else
            config="flue/examples/cls_music_lr5e6_xlm_base_cased.cfg"
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
        python flue/accuracy_from_task1.py --logits_file ./flue/experiments/cls_music_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.29 --labels_file ./flue/data/cls/processed/music/test.label
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
        else
            config="flue/examples/cls_DVD_lr5e6_xlm_base_cased.cfg"
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
        python flue/accuracy_from_task1.py --logits_file ./flue/experiments/cls_dvd_xlm_base_cased/bs_8_dropout_0.1_ep_30_lre_5e6_lrp_5e6/test.pred.29 --labels_file ./flue/data/cls/processed/dvd/test.label
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
            echo "Installation de MLflow pour le suivi des expériences..."
            pip install mlflow
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
            echo "Vérification de l'installation de MLflow..."
            if ! command -v mlflow &> /dev/null; then
                echo "⚠️  MLflow n'est pas installé. Installation automatique..."
                pip install mlflow
            else
                echo "✅ MLflow est déjà installé."
            fi
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
        else
            config="flue/examples/cls_books_lr5e6_hf_base_uncased.cfg"  # configuration par défaut
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/prepare-data-cls.sh ./flue/extract_split_cls.py ./flue/binarize.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_from_hf.py ./flue/train_with_mlflow.py
        echo "Récupération des données CLS..."
        if [ ! -f "$DATA_DIR/cls/raw/cls-acl10-unprocessed.tar.gz" ]; then
            echo "Vous devez faire une demande pour les données à l'adresse https://zenodo.org/record/3251672"
            echo "et placer le fichier dans $DATA_DIR/cls/raw/cls-acl10-unprocessed.tar"
            exit 1
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
        echo "Lancement de l'évaluation CLS books avec MLflow tracking..."
        export MODEL_NAME
        source $config
        
        # Generate unique experiment name with timestamp
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        EXPERIMENT_NAME="FLUE-CLS-Books-${MODEL_NAME}-${TIMESTAMP}"
        
        echo "Experiment MLflow: $EXPERIMENT_NAME"
        echo "Modèle: $model_name_or_path"
        echo "Dossier de sortie: $output_dir"
        
        python flue/train_with_mlflow.py \
                                        --train_file $data_dir/train.csv \
                                        --validation_file $data_dir/valid.csv \
                                        --test_file $data_dir/test.csv \
                                        --model_name_or_path $model_name_or_path \
                                        --output_dir $output_dir \
                                        --max_seq_length 512 \
                                        --learning_rate $lr \
                                        --num_train_epochs $epochs \
                                        --save_steps $save_steps \
                                        --per_device_train_batch_size $batch_size \
                                        --per_device_eval_batch_size $batch_size \
                                        --task_name "cls-books" \
                                        --experiment_name "$EXPERIMENT_NAME" \
                                        |& tee output.log
        echo "Calcul de la précision à partir des prédictions Hugging Face..."
        echo "Précision de validation à partir de l'entraînement:"
            python -c "import json; data=json.load(open('$output_dir/eval_results.json')); print(f\"Précision de validation: {data['eval_accuracy']*100:.2f}% sur {data['eval_samples']} exemples\")"
        echo "Précision de test à partir des prédictions:"
            python flue/accuracy_from_hf.py --predictions_file $output_dir/predict_results_None.txt --labels_file $DATA_DIR/cls/processed/books/test.label
        echo ""
        echo "=== MLflow Tracking Information ==="
        echo "Les résultats de l'expérience ont été enregistrés dans MLflow."
        echo "Pour visualiser les résultats, lancez: mlflow ui --backend-store-uri ./mlflow_logs"
        echo "Puis ouvrez http://localhost:5000 dans votre navigateur."
        echo "Nom de l'expérience: $EXPERIMENT_NAME"
        ;;
    pawsx)
        if [ -z "$INSTALL_LIBS" ]; then
            echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
            exit 1
        fi
        if [ $INSTALL_LIBS == true ]; then
            echo "Installation des librairies requises..."
            pip install -r ./requirements.txt
            echo "Librairies installées."
        else
            echo "Installation des librairies ignorée."
        fi
        
        if [ ! -z "$CUSTOM_CONFIG" ]; then
            config="flue/examples/$CUSTOM_CONFIG"
        else
            config="flue/examples/pawsx_lr5e6_xlm_base_cased.cfg"
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Récupération des données PAWSX..."
        ./flue/get-data-xnli.sh $DATA_DIR
        echo "Préparation des données PAWSX..."
        ./flue/prepare-data-pawsx.sh $DATA_DIR $MODEL_PATH false
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
        else
            config="flue/examples/xnli_lr5e6_xlm_base_cased.cfg"
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
        python ./flue/accuracy_from_task3.py
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
        else
            config="flue/examples/xnli_lr5e6_hf_base_uncased.cfg"  # configuration par défaut
        fi
        echo "Utilisation de la configuration: $config"
        
        echo "Ajout des droits d'exécution aux scripts..."
        chmod +x ./flue/extract_xnli.py ./flue/binarize.py ./flue/data/hg_data_tsv_to_csv.py
        chmod +x ./flue/accuracy_from_hf.py

        echo "Récupération des données XNLI..."
        ./flue/get-data-xnli.sh $DATA_DIR/xnli

        echo "Préparation des données XNLI..."
        python flue/extract_xnli.py --indir $DATA_DIR/xnli/processed \
                                 --outdir $DATA_DIR/xnli/processed \
                                 --do_lower false
        echo "Conversion des fichiers TSV au format CSV..."
        python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/xnli/processed/
        
        echo "Lancement de l'évaluation XNLI..."
        export MODEL_NAME
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
                                        --trust_remote_code \
                                        --overwrite_output_dir \
                                        |& tee output.log
        echo "Calcul de la précision à partir des prédictions Hugging Face..."
        echo "Précision de validation à partir de l'entraînement:"
            python -c "import json; data=json.load(open('$output_dir/eval_results.json')); print(f\"Précision de validation: {data['eval_accuracy']*100:.2f}% sur {data['eval_samples']} exemples\")"
        echo "Précision de test à partir des prédictions:"
            python flue/accuracy_from_hf.py --predictions_file $output_dir/predict_results_None.txt --labels_file $DATA_DIR/xnli/processed/test.label
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
            echo "Configuration personnalisée: $CUSTOM_CONFIG (non utilisée pour WSD)"
        fi
        echo "Modèle utilisé: $MODEL_NAME"
        
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
        
        # Création du dossier d'expérience si nécessaire
        mkdir -p ./flue/experiments/wsd_${MODEL_NAME}
        
        # Lancement de l'évaluation WSD avec le modèle spécifié
        echo "Lancement de l'évaluation WSD avec le modèle $MODEL_NAME..."
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
        
        python flue_vsd.py --exp_name wsd_${MODEL_NAME}_evaluation \
                          --model $MODEL_NAME \
                          --data ../../../$DATA_DIR/wsd/processed \
                          --padding 80 \
                          --batchsize 32 \
                          --device $DEVICE \
                          --output ../../../flue/experiments/wsd_${MODEL_NAME} \
                          --output_logs ../../../flue/experiments/wsd_${MODEL_NAME}/evaluation_logs.csv \
                          --output_pred ../../../flue/experiments/wsd_${MODEL_NAME}/predictions.txt \
                          --output_score ../../../flue/experiments/wsd_${MODEL_NAME}/scores.csv
        cd ../../..
        
        echo "Évaluation WSD terminée."
        echo "Résultats disponibles dans: ./flue/experiments/wsd_${MODEL_NAME}/"
        if [ -f "./flue/experiments/wsd_${MODEL_NAME}/scores.csv" ]; then
            echo "Scores d'évaluation:"
            cat ./flue/experiments/wsd_${MODEL_NAME}/scores.csv
        fi
        ;;
    *)
        echo "Veuiller spécifier une tache valide."
        echo "Tâches valides: cls-books-XLM, cls-music-XLM, cls-dvd-XLM, cls-HF, xnli-HF, xnli-XLM, pawsx, parse, wsd"
        exit 1
esac
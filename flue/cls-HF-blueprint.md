# Blueprint: cls-HF Case

## Vue d'ensemble
Le cas `cls-HF` exécute une évaluation de classification de sentiment cross-lingue en utilisant le framework Hugging Face Transformers.

## Paramètres d'entrée
- **Tâche**: `cls-HF`
- **Installation des librairies**: `true`/`false`
- **Nom du modèle**: Par défaut `flaubert_base_cased`, ou modèle personnalisé
- **Fichier de configuration**: Configuration personnalisée ou par défaut

## Flux d'exécution

### 1. Validation des paramètres
```bash
if [ -z "$INSTALL_LIBS" ]; then
    echo "Veuillez spécifier si les librairies doivent être installées (true/false)."
    exit 1
fi
```

### 2. Installation des dépendances (optionnelle)
- **Si `INSTALL_LIBS == true`**:
  - Installe les librairies depuis `./libraries/hg-requirements.txt`
  - Inclut: transformers, datasets, torch, pandas, scikit-learn
- **Si `INSTALL_LIBS == false`**:
  - Ignore l'installation

### 3. Sélection de la configuration
```bash
if [ ! -z "$CUSTOM_CONFIG" ]; then
    config="flue/examples/$CUSTOM_CONFIG"
else
    config="flue/examples/cls_books_lr5e6_hf_base_uncased.cfg"  # par défaut
fi
```

### 4. Préparation de l'environnement
- Attribution des droits d'exécution aux scripts:
  - `./flue/prepare-data-cls.sh`
  - `./flue/extract_split_cls.py`
  - `./flue/binarize.py`
  - `./flue/data/hg_data_tsv_to_csv.py`
  - `./flue/accuracy_from_hf.py`

### 5. Gestion des données CLS
#### 5.1 Vérification des données
- Vérifie la présence de `cls-acl10-unprocessed.tar.gz` dans `flue/data/cls/raw/`
- **Si absent**: Affiche message d'erreur et lien Zenodo
- **Si présent**: Décompresse automatiquement

#### 5.2 Préparation des données
1. **Extraction et division**:
   ```bash
   python flue/extract_split_cls.py --indir $DATA_DIR/cls/raw/cls-acl10-unprocessed \
                            --outdir $DATA_DIR/cls/processed \
                            --do_lower false \
                            --use_hugging_face true
   ```

2. **Conversion de format**:
   ```bash
   python flue/data/hg_data_tsv_to_csv.py $DATA_DIR/cls/processed/books/
   ```
   - Convertit les fichiers TSV vers CSV pour compatibilité Hugging Face

### 6. Exécution de l'évaluation
#### 6.1 Configuration de l'environnement
- Exporte `MODEL_NAME` comme variable d'environnement
- Source le fichier de configuration sélectionné

#### 6.2 Lancement de l'entraînement/évaluation
```bash
python tools/transformers/examples/pytorch/text-classification/run_glue.py \
    --train_file $data_dir/train.csv \
    --validation_file $data_dir/valid.csv \
    --test_file $data_dir/test.csv \
    --model_name_or_path $model_name_or_path \
    --output_dir $output_dir \
    --max_seq_length 512 \
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate $lr \
    --num_train_epochs $epochs \
    --save_steps $save_steps \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size $batch_size \
    --overwrite_output_dir
```

### 7. Calcul des résultats
#### 7.1 Précision de validation
```bash
python -c "import json; data=json.load(open('$output_dir/eval_results.json')); 
           print(f\"Précision de validation: {data['eval_accuracy']*100:.2f}% sur {data['eval_samples']} exemples\")"
```

#### 7.2 Précision de test
```bash
python flue/accuracy_from_hf.py --predictions_file $output_dir/predict_results_None.txt \
                                --labels_file $DATA_DIR/cls/processed/books/test.label
```

## Fichiers impliqués

### Données d'entrée
- `flue/data/cls/raw/cls-acl10-unprocessed.tar.gz` (manuel)
- `flue/data/cls/processed/books/` (généré automatiquement)

### Configuration
- `flue/examples/cls_books_lr5e6_hf_base_uncased.cfg` (par défaut)
- `flue/examples/$CUSTOM_CONFIG` (personnalisé)

### Scripts utilisés
- `flue/extract_split_cls.py`: Extraction et division des données
- `flue/data/hg_data_tsv_to_csv.py`: Conversion de format
- `tools/transformers/examples/pytorch/text-classification/run_glue.py`: Entraînement HF
- `flue/accuracy_from_hf.py`: Calcul de précision

### Fichiers de sortie
- `$output_dir/eval_results.json`: Résultats de validation
- `$output_dir/predict_results_None.txt`: Prédictions de test
- `output.log`: Journal d'exécution

## Variables de configuration utilisées
Depuis le fichier `.cfg`:
- `$data_dir`: Répertoire des données
- `$model_name_or_path`: Chemin vers le modèle
- `$output_dir`: Répertoire de sortie
- `$lr`: Taux d'apprentissage
- `$epochs`: Nombre d'époques
- `$batch_size`: Taille du batch
- `$save_steps`: Fréquence de sauvegarde

## Particularités
1. **Format Hugging Face**: Utilise le format CSV au lieu de TSV
2. **Variable d'environnement**: `MODEL_NAME` est exportée pour utilisation dans la config
3. **Logs**: Sortie sauvegardée dans `output.log` avec `tee`
4. **Évaluation automatique**: Calcule automatiquement les métriques de validation et test
5. **Modularité**: Support de configurations personnalisées via le 4ème paramètre

## Prérequis
1. **Données**: Téléchargement manuel du dataset CLS depuis Zenodo
2. **Modèle**: Présence du modèle dans `flue/pretrained_models/`
3. **Dépendances**: Installation via `libraries/hg-requirements.txt`
4. **Répertoire**: Exécution depuis le dossier racine FLUE

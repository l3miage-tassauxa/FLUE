# Framework d'Évaluation FLUE

FLUE (French Language Understanding Evaluation) est un framework d'évaluation complet pour les modèles de langue française. Ce guide explique comment utiliser le script `evaluation_auto.sh` pour évaluer vos modèles sur diverses tâches de TAL français.

# Note
/!\ ce dépôt est encore en construction, certaines fonctionnalités ne sont pas encore implémentées ou risquent de ne pas correctement fonctionner.

## Prérequis
1) cloner le dépôt transformers https://github.com/formiel/transformers et le placer dans le répertoire `FLUE/tools`

2) Avant d'exécuter un premier fine-tuning, il faut créer un nouvel environnement python, et installer la version de PyTorch compatible avec sa carte graphique.

Pour cela, il existe un tableau qui référence les cartes graphiques avec leur capacité de calcul:
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus#compute)
- [Legacy CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-legacy-gpus)
- [Installation de PyTorch + CUDA](https://pytorch.org/get-started/locally/)

Par exemple, avec PyTorch 2.7.1 sont prises en charge les cartes graphiques avec une capacité de calcul comprise entre 5.0 et 9.0

```bash
python -c "import torch; print(torch.__version__)"

2.7.1+cu126

python -c "import torch; print(torch.cuda.get_arch_list())"

['sm_50', 'sm_60', 'sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90']
```

Une fois l'installation terminée, il est possible de lancer `evaluation_auto.sh` depuis le répertoire racine du projet `FLUE`

## Démarrage Rapide


1. **Placer votre modèle** dans `flue/pretrained_models/<nom_de_votre_modele>`

2. **Démarrer le fine-tuning d'une tâche**

se placer dans le répertoire `FLUE` et lancer:
   ```bash
   ./flue/evaluation_auto.sh <tâche> <installer_libs> <fichier_config>
   ```

## Utilisation

### Structure de Commande

```bash
bash ./flue/evaluation_auto.sh <tache> <installer_libs> <fichier_config>
```

**Les trois paramètres sont obligatoires :**
- `<tache>` : La tâche d'évaluation à exécuter
- `<installer_libs>` : Si installer les dépendances (`true`/`false`)
- `<fichier_config>` : Nom du fichier de configuration (doit exister dans `flue/examples/`)

### Obtenir de l'Aide

```bash
bash ./flue/evaluation_auto.sh --help
# ou
bash ./flue/evaluation_auto.sh -h
```

### Tâches Disponibles sur `evaluation_auto.sh`

#### Tâches avec la librairie Hugging Face Transformers (Recommandées)
- **`cls-HF`** : Analyse de sentiment cross-lingue avec Hugging Face Transformers
- **`xnli-HF`** : Inférence en langage naturel cross-lingue avec Hugging Face Transformers

#### Tâches avec la librairie XLM (Dépassé)
- **`cls-XLM`** : Analyse de sentiment cross-lingue avec le framework XLM
- **`xnli-XLM`** : Inférence en langage naturel cross-lingue avec le framework XLM
- **`pawsx`** : Paraphrase Adversaries from Word Scrambling for Cross-lingual Understanding

#### Tâches Hugging Face avec suivi sur Mlflow
- **`mlflow-cls-books-HF`**
- **`mlflow-cls-music-HF`** 
- **`mlflow-cls-dvd-HF`** 
- **`mlflow-pawsx-HF`**


## Exemples

### 1. Évaluer avec le Modèle par Défaut

```bash
# Utiliser le modèle flaubert_base_cased par défaut, installer les librairies
bash ./flue/evaluation_auto.sh xnli-HF true

# Utiliser le modèle par défaut, ignorer l'installation des librairies
bash ./flue/evaluation_auto.sh cls-HF false
```

### 2. Évaluer avec Votre Propre Modèle

```bash
# Évaluer votre modèle personnalisé
bash ./flue/evaluation_auto.sh xnli-HF true mon_modele_francais

# Évaluer CamemBERT
bash ./flue/evaluation_auto.sh cls-HF false camembert-base
```

### 3. Utiliser une Configuration Personnalisée

```bash
# Utiliser votre propre fichier de configuration
bash ./flue/evaluation_auto.sh xnli-HF true mon_modele chemin/vers/ma_config.cfg
```

## Configuration des Modèles

### Structure des Répertoires

Placez vos modèles dans le répertoire `flue/pretrained_models/` :

```
flue/pretrained_models/
├── flaubert_base_cased/          # Modèle par défaut
├── mon_modele_francais/          # Votre modèle personnalisé
├── camembert-base/               # CamemBERT
└── nom_de_votre_modele/          # Tout autre modèle
    ├── config.json
    ├── pytorch_model.bin (ou model.safetensors)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

## Fichiers de Configuration

### Configurations par Défaut

Le framework inclut des configurations par défaut dans `flue/examples/` :
- `xnli_lr5e6_hf_base_uncased.cfg` - Configuration XNLI par défaut
- `cls_books_lr5e6_hf_base_uncased.cfg` - Configuration CLS par défaut
- `xnli_lr5e6_xlm_base_cased.cfg` - Configuration XNLI XLM
- `pawsx_lr5e6_xlm_base_cased.cfg` - Configuration PAWSX

### Configuration Personnalisée

Créez votre propre fichier `.cfg` avec ces paramètres :

```bash
# Paramètres du modèle
model_type=flaubert
model_name=mon_modele
model_name_or_path=flue/pretrained_models/mon_modele

# Paramètres d'entraînement
batch_size=8
lr=0.000005
epochs=10
dropout=0.1

# Chemins des données
data_dir=flue/data/xnli/processed-csv
train_file=flue/data/xnli/processed-csv/train.csv
validation_file=flue/data/xnli/processed-csv/valid.csv
test_file=flue/data/xnli/processed-csv/test.csv

# Sortie
output_dir=flue/experiments/mon_modele/results
max_seq_length=512
```

## Exigences des Données

Les tâches nécessitent différents ensembles de données :

### Tâches de Classification (CLS)
- **Localisation** : `flue/data/cls/`
- **Fichier requis** : `cls-acl10-unprocessed.tar.gz`
- **Téléchargement** : Disponible sur Zenodo (voir documentation principale)

### Tâches XNLI
- **Localisation** : `flue/data/xnli/`
- **Fichiers requis** : Données XNLI traduites en français
- **Format** : Fichiers CSV traités

### Tâches PAWSX
- **Localisation** : `flue/data/pawsx/`
- **Fichiers requis** : Données PAWSX françaises

### Tâches de Parsing/WSD
- **Statut** : Non implémentées (en développement)

## Résultats

### Emplacement de Sortie

Les résultats sont sauvegardés dans : `flue/experiments/<type_modele>/<nom_exp>/<id_exp>/`

### Fichiers de Résultats

- `eval_results.json` : Précision de validation et métriques
- `predict_results_None.txt` : Prédictions de test
- `training_logs/` : Journaux de progression d'entraînement
- Points de contrôle du modèle (si activés)

### Calcul de la Précision

Le framework calcule et affiche automatiquement :
- **Précision de validation** à partir des journaux d'entraînement
- **Précision de test** à partir des prédictions vs. étiquettes de référence

## Dépannage

### Problèmes Courants

1. **Arguments insuffisants**
   ```
   Usage: bash ./flue/evaluation_auto.sh <task> <install_libs> <config_file>
   ```
   **Solution** : Fournissez les trois paramètres obligatoires

2. **Tâche invalide**
   ```
   Please specify a valid task.
   ```
   **Solution** : Utilisez `--help` pour voir les tâches disponibles

3. **Fichier de configuration non trouvé**
   ```
   Configuration file 'mon_config.cfg' not found
   ```
   **Solution** : Vérifiez que le fichier existe dans `flue/examples/`

4. **Données non trouvées (CLS)**
   ```
   Error: cls-acl10-unprocessed.tar.gz not found
   ```
   **Solution** : Téléchargez les données CLS depuis Zenodo

5. **Permission refusée**
   ```
   Error: Permission denied
   ```
   **Solution** : Exécutez `chmod +x ./flue/evaluation_auto.sh`

6. **Tâches non implémentées**
   ```
   task not yet implemented...
   ```
   **Solution** : Ces tâches sont en développement

### Dépendances

Installez les librairies requises en définissant le deuxième paramètre à `true` :
```bash
bash ./flue/evaluation_auto.sh ma_tache true mon_config.cfg
```

Le script installera automatiquement :
- Dépendances XLM (pour les tâches XLM)
- Dépendances Hugging Face (pour les tâches HF)

## Contribution

### Ajouter une Nouvelle Tâche

Pour implémenter une nouvelle tâche dans `evaluation_auto.sh` :

1. **Ajouter le cas dans le switch** :
```bash
export MODEL_NAME=mon_modele_personnalise
export BATCH_SIZE=16
bash ./flue/evaluation_auto.sh xnli-HF false
```

### Métriques d'Évaluation Personnalisées

Ajoutez vos propres scripts d'évaluation en suivant le modèle de :
- `flue/accuracy_from_hf.py` - Traitement des résultats Hugging Face
- `flue/accuracy_from_task3.py` - Traitement des résultats XLM

### Validation Modulaire des Arguments

Le script utilise maintenant une approche modulaire pour la validation des arguments :
- Chaque tâche valide ses propres paramètres requis
- La validation `INSTALL_LIBS` se fait au niveau de chaque tâche
- Cela améliore la maintenabilité et la clarté du code

## Contribution

Pour ajouter de nouvelles tâches ou modèles :
1. Créez des fichiers de configuration dans `flue/examples/`
2. Ajoutez la gestion des cas dans `evaluation_auto.sh`
3. Implémentez le prétraitement des données si nécessaire
4. Ajoutez des scripts de traitement des résultats

## Licence

Notre contribution est basée sur le benchmark FLUE original. Veuillez citer l'article lors de l'utilisation de ce framework d'évaluation.

# XLM

# XLM

./flue/evaluation_auto.sh cls-books-XLM false cls_books_lr5e6_xlm_base_cased.cfg

./flue/evaluation_auto.sh cls-music-XLM false cls_music_lr5e6_xlm_base_cased.cfg

./flue/evaluation_auto.sh cls-dvd-XLM false cls_dvd_lr5e6_xlm_base_cased.cfg

./flue/evaluation_auto.sh pawsx-XLM false pawsx_lr5e6_xlm_base_cased.cfg

./flue/evaluation_auto.sh xnli-XLM false xnli_lr5e6_xlm_base_cased.cfg

# HuggingFace

./flue/evaluation_auto.sh cls-books-HF false cls_books_lr5e6_hf.cfg

./flue/evaluation_auto.sh cls-music-HF false cls_music_lr5e6_hf.cfg

./flue/evaluation_auto.sh cls-dvd-HF false cls_dvd_lr5e6_hf.cfg

./flue/evaluation_auto.sh pawsx-HF false pawsx_lr5e6_hf.cfg

# Mlflow

mlflow server --host 0.0.0.0 --port 5000

./flue/evaluation_auto.sh mlflow-cls-books-HF false cls_books_lr5e6_hf.cfg

./flue/evaluation_auto.sh mlflow-cls-music-HF false cls_books_lr5e6_hf.cfg

./flue/evaluation_auto.sh mlflow-cls-dvd-HF false cls_dvd_lr5e6_hf.cfg

./flue/evaluation_auto.sh mlflow-pawsx-HF false pawsx_lr5e6_hf.cfg


# Lancer une tâche de type mlflow:

1) une fois connecté à une session ssh (ex: decore1), démarrer le tracking server depuis le répertoire `FLUE`:
mlflow server --host 0.0.0.0 --port 5000

L'interface deviens accessible via http://localhost:5000/ . Au début, aucune expérience n'est référencée. C'est normal puisque les résultats sont stockés localement (dans le dossier `FLUE/mlruns`).

*NB: piste à explorer, utiliser DataBricks pour héberger les résultats d'évaluation dans le cloud*

Une fois une première tâche de type mlflow-<nom_tâche>-HF lancée, le nom de l'expérience apparaît à gauche, avec les runs associées sur la droite. (Ajouter images ?)

Chaque expérience est composée de "runs", dont chaque couple de run correspond à une tentative de fine tuning. La première run (ex: flaubert_basecased<date>_<heure>) recensera les hyperparamètres et le suivi des métriques d'entraînement, tandis que la seconde, au nom aléatoire, contiendra uniquement les dernières valeur de la phase d'évaluation. 
Pour l'instant ce comportement par défaut n'a pas pu être changé.

## Exemples de Commandes par Type de Tâche

### Tâches XLM
```bash
# Classification de sentiments - Livres
./flue/evaluation_auto.sh cls-books-XLM false cls_books_lr5e6_xlm_base_cased.cfg

# Classification de sentiments - Musique  
./flue/evaluation_auto.sh cls-music-XLM false cls_music_lr5e6_xlm_base_cased.cfg

# Classification de sentiments - DVD
./flue/evaluation_auto.sh cls-dvd-XLM false cls_dvd_lr5e6_xlm_base_cased.cfg

# Identification de paraphrases
./flue/evaluation_auto.sh pawsx-XLM false pawsx_lr5e6_xlm_base_cased.cfg

# Inférence en langage naturel
./flue/evaluation_auto.sh xnli-XLM false xnli_lr5e6_xlm_base_cased.cfg
```

### Tâches Hugging Face
```bash
# Classification de sentiments - Livres
./flue/evaluation_auto.sh cls-books-HF false cls_books_lr5e6_hf.cfg

# Classification de sentiments - Musique
./flue/evaluation_auto.sh cls-music-HF false cls_music_lr5e6_hf.cfg

# Classification de sentiments - DVD
./flue/evaluation_auto.sh cls-dvd-HF false cls_dvd_lr5e6_hf.cfg

# Identification de paraphrases
./flue/evaluation_auto.sh pawsx-HF false pawsx_lr5e6_hf.cfg
```

### Tâches avec Suivi MLflow

#### Configuration du Serveur MLflow
Avant de lancer les tâches MLflow, démarrez le serveur de suivi :
```bash
mlflow server --host 0.0.0.0 --port 5000
```

#### Commandes d'Évaluation MLflow
```bash
# Classification de sentiments avec suivi MLflow - Livres
./flue/evaluation_auto.sh mlflow-cls-books-HF false cls_books_lr5e6_hf.cfg

# Classification de sentiments avec suivi MLflow - Musique
./flue/evaluation_auto.sh mlflow-cls-music-HF false cls_music_lr5e6_hf.cfg

# Classification de sentiments avec suivi MLflow - DVD
./flue/evaluation_auto.sh mlflow-cls-dvd-HF false cls_dvd_lr5e6_hf.cfg

# Identification de paraphrases avec suivi MLflow
./flue/evaluation_auto.sh mlflow-pawsx-HF false pawsx_lr5e6_hf.cfg
```

## Guide d'Utilisation MLflow

### Démarrage du Serveur de Suivi

1. **Connexion SSH** : Connectez-vous à votre session SSH (ex: decore1)
2. **Démarrage du serveur** : Depuis le répertoire `FLUE`, exécutez :
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```
3. **Accès à l'interface** : L'interface devient accessible via http://localhost:5000/

### Fonctionnement du Suivi MLflow

- **Stockage local** : Les résultats sont stockés localement dans le dossier `FLUE/mlruns`
- **Interface vide au début** : Aucune expérience n'apparaît initialement, c'est normal
- **Piste future** : Exploration possible de DataBricks pour l'hébergement cloud des résultats

### Structure des Expériences

Une fois qu'une tâche `mlflow-<nom_tâche>-HF` est lancée :

- **Organisation** : Le nom de l'expérience apparaît à gauche, avec les runs associées à droite
- **Composition des runs** : Chaque expérience contient des "runs" organisées par paires
  - **Premier run** : `flaubert_basecased<date>_<heure>` - contient les hyperparamètres et métriques d'entraînement
  - **Second run** : Nom aléatoire - contient uniquement les valeurs finales de la phase d'évaluation

**Note** : Ce comportement par défaut n'a pas encore pu être modifié.

# HuggingFace

./flue/evaluation_auto.sh cls-books-HF false cls_books_lr5e6_hf.cfg

./flue/evaluation_auto.sh cls-music-HF false cls_music_lr5e6_hf.cfg

./flue/evaluation_auto.sh cls-dvd-HF false cls_dvd_lr5e6_hf.cfg

./flue/evaluation_auto.sh pawsx-HF false pawsx_lr5e6_hf.cfg

# Mlflow

mlflow server --host 0.0.0.0 --port 5000

./flue/evaluation_auto.sh mlflow-cls-books-HF false cls_books_lr5e6_hf.cfg

./flue/evaluation_auto.sh mlflow-cls-music-HF false cls_music_lr5e6_hf.cfg

./flue/evaluation_auto.sh mlflow-cls-dvd-HF false cls_dvd_lr5e6_hf.cfg

./flue/evaluation_auto.sh mlflow-pawsx-HF false pawsx_lr5e6_hf.cfg


# Lancer une tâche de type mlflow:

1) une fois connecté à une session ssh (ex: decore1), démarrer le tracking server depuis le répertoire `FLUE`:
mlflow server --host 0.0.0.0 --port 5000

L'interface deviens accessible via http://localhost:5000/ . Au début, aucune expérience n'est référencée. C'est normal puisque les résultats sont stockés localement (dans le dossier `FLUE/mlruns`).

*NB: piste à explorer, utiliser DataBricks pour héberger les résultats d'évaluation dans le cloud*

Une fois une première tâche de type mlflow-<nom_tâche>-HF lancée, le nom de l'expérience apparaît à gauche, avec les runs associées sur la droite. (Ajouter images ?)

Chaque expérience est composée de “runs”, dont chaque couple de run correspond à une tentative de fine tuning. La première run (ex: flaubert_basecased<date>_<heure>) recensera les hyperparamètres et le suivi des métriques d’entraînement, tandis que la seconde, au nom aléatoire, contiendra uniquement les dernières valeur de la phase d'évaluation. 
Pour l'instant ce comportement par défaut n'a pas pu être changé.
# Framework d'Évaluation FLUE

FLUE (French Language Understanding Evaluation) est un framework d'évaluation complet pour les modèles de langue française. Ce guide explique comment utiliser le script `evaluation_auto.sh` pour évaluer vos modèles sur diverses tâches de TAL français.

## Démarrage Rapide

1. **Cloner le dépôt**
2. **Placer votre modèle** dans `flue/pretrained_models/nom_de_votre_modele/`
3. **Lancer l'évaluation** :
   ```bash
   bash ./flue/evaluation_auto.sh <tache> <installer_libs> [nom_modele] [fichier_config]
   ```

## Utilisation

### Structure de Commande de Base

```bash
bash ./flue/evaluation_auto.sh <tache> <installer_libs> [nom_modele] [fichier_config]
```

**Paramètres :**
- `<tache>` : Obligatoire. La tâche d'évaluation à exécuter
- `<installer_libs>` : Obligatoire. Si installer les dépendances (`true`/`false`)
- `[nom_modele]` : Optionnel. Le nom du répertoire de votre modèle (défaut : `flaubert_base_cased`)
- `[fichier_config]` : Optionnel. Chemin vers un fichier de configuration personnalisé

### Tâches Disponibles

#### Tâches Hugging Face avec Intégration MLflow
- **`cls-books-HF`** : Classification de sentiment sur les livres avec Hugging Face Transformers et tracking MLflow
- **`cls-music-HF`** : Classification de sentiment sur la musique avec Hugging Face Transformers et tracking MLflow  
- **`cls-dvd-HF`** : Classification de sentiment sur les DVD avec Hugging Face Transformers et tracking MLflow
- **`pawsx-HF`** : Paraphrase Adversaries from Word Scrambling avec Hugging Face Transformers et tracking MLflow
- **`xnli-HF`** : Inférence en langage naturel cross-lingue avec Hugging Face Transformers et tracking MLflow

#### Tâches XLM (Héritées)
- **`cls-XLM`** : Analyse de sentiment cross-lingue avec le framework XLM
- **`xnli-XLM`** : Inférence en langage naturel cross-lingue avec le framework XLM
- **`pawsx`** : Paraphrase Adversaries from Word Scrambling pour la compréhension cross-lingue

## Intégration MLflow

Toutes les tâches Hugging Face incluent un tracking automatique des expériences avec MLflow :

### Démarrage Rapide MLflow
1. **Lancer une expérience** : `bash tache.sh cls-books-HF`
2. **Démarrer l'interface MLflow** : `bash start_mlflow_ui.sh`
3. **Accéder au dashboard** : http://localhost:5000

### Expériences Trackées
- **FLUE_CLS_Books_HF** : Classification de sentiment sur les livres
- **FLUE_CLS_Music** : Classification de sentiment sur la musique
- **FLUE_CLS_DVD** : Classification de sentiment sur les DVD
- **FLUE_PAWSX** : Tâche PAWSX avec Hugging Face
- **FLUE_XNLI** : Tâche XNLI avec Hugging Face

### Métriques Automatiques
- Accuracy de validation et test
- Loss d'entraînement
- Temps d'exécution
- Paramètres d'hyperparamètres
- Artéfacts de modèle et fichiers de configuration

## Exemples

### 1. Évaluer avec le Modèle par Défaut

```bash
# Utiliser le modèle flaubert_base_cased par défaut, installer les librairies
bash ./flue/evaluation_auto.sh xnli-HF true

# Utiliser le modèle par défaut, ignorer l'installation des librairies
bash ./flue/evaluation_auto.sh cls-books-HF false

# Lancer une évaluation avec tracking MLflow
bash tache.sh cls-music-HF
```

### 2. Évaluer avec Votre Propre Modèle

```bash
# Évaluer votre modèle personnalisé
bash ./flue/evaluation_auto.sh xnli-HF true mon_modele_francais

# Évaluer CamemBERT avec MLflow tracking
bash tache.sh cls-books-HF camembert-base

# Évaluer avec une tâche spécifique
bash ./flue/evaluation_auto.sh cls-dvd-HF false camembert-base
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

### Types de Modèles Supportés

- **FlauBERT** : `flaubert_base_cased`, `flaubert_base_uncased`
- **CamemBERT** : `camembert-base`, `camembert-large`
- **Modèles personnalisés** : Tout modèle français compatible Hugging Face
- **Modèles fine-tunés** : Vos propres versions fine-tunées

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

### Tâche XNLI
- **Automatique** : Les données sont téléchargées automatiquement depuis le dataset XNLI de Facebook
- **Aucune configuration manuelle requise**

### Tâche CLS
1. **Téléchargement manuel requis** : Visitez [https://zenodo.org/record/3251672](https://zenodo.org/record/3251672)
2. **Demandez l'accès** au dataset CLS
3. **Placez le fichier** : `cls-acl10-unprocessed.tar.gz` dans `flue/data/cls/raw/`

### Tâche PAWSX
- **Automatique** : Les données sont téléchargées automatiquement
- **Aucune configuration manuelle requise**

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

1. **Modèle non trouvé**
   ```
   Erreur : Modèle 'mon_modele' non trouvé dans flue/pretrained_models/
   ```
   **Solution** : Assurez-vous que le répertoire de votre modèle existe avec tous les fichiers requis

2. **Données non trouvées (CLS)**
   ```
   Erreur : cls-acl10-unprocessed.tar.gz non trouvé
   ```
   **Solution** : Téléchargez les données CLS depuis Zenodo (voir Exigences des Données)

3. **Problèmes de mémoire GPU**
   - Réduisez `batch_size` dans votre fichier de configuration
   - Utilisez un modèle plus petit
   - Réduisez `max_seq_length`

4. **Permission refusée**
   ```
   Erreur : Permission refusée
   ```
   **Solution** : Exécutez `chmod +x ./flue/evaluation_auto.sh`

5. **Paramètre manquant**
   ```
   Veuillez spécifier si les librairies doivent être installées (true/false).
   ```
   **Solution** : Le script valide maintenant les paramètres pour chaque tâche - assurez-vous de fournir tous les arguments requis

### Dépendances

Installez les librairies requises en définissant le deuxième paramètre à `true` :
```bash
bash ./flue/evaluation_auto.sh xnli-HF true
```

Cela installe :
- transformers
- datasets
- torch
- pandas
- scikit-learn
- Autres dépendances depuis `libraries/hg-requirements.txt`

## Utilisation Avancée

### Variables d'Environnement

Vous pouvez surcharger les paramètres de configuration via les variables d'environnement :
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

Ce framework est basé sur le benchmark FLUE original. Veuillez citer l'article original lors de l'utilisation de ce framework d'évaluation.
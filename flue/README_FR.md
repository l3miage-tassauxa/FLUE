# Framework d'Évaluation FLUE

FLUE (French Language Understanding Evaluation) est un framework d'évaluation complet pour les modèles de langue française. Ce guide explique comment utiliser le script `evaluation_auto.sh` pour évaluer vos modèles sur diverses tâches de TAL français.

## Démarrage Rapide

1. **Cloner le dépôt**
2. **S'assurer d'être dans le répertoire racine FLUE**
3. **Lancer l'évaluation** :
   ```bash
   bash ./flue/evaluation_auto.sh <tache> <installer_libs> <fichier_config>
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

### Tâches Disponibles

#### Tâches XLM
- `cls-books-XLM` : Classification de sentiments - livres
- `cls-music-XLM` : Classification de sentiments - musique
- `cls-dvd-XLM` : Classification de sentiments - DVD
- `xnli-XLM` : Inférence en langage naturel cross-lingue
- `pawsx-XLM` : Identification de paraphrases

#### Tâches Hugging Face
- `cls-books-HF` : Classification livres - HF
- `cls-music-HF` : Classification musique - HF
- `cls-dvd-HF` : Classification DVD - HF
- `pawsx-HF` : PAWSX avec Hugging Face

#### Tâches MLflow
- `mlflow-cls-books-HF` : Classification livres avec suivi MLflow
- `mlflow-cls-music-HF` : Classification musique avec suivi MLflow
- `mlflow-cls-dvd-HF` : Classification DVD avec suivi MLflow
- `mlflow-pawsx-HF` : PAWSX avec suivi MLflow

#### Tâches Non Implémentées
- `xnli-HF` : XNLI avec intégration Hugging Face (à implémenter)
- `parsing` : Analyse syntaxique (à implémenter)
- `wsd` : Désambiguïsation lexicale (à implémenter)

## Exemples

### Classification de Sentiments (Books)
```bash
bash ./flue/evaluation_auto.sh cls-books-XLM true cls_books_lr5e6_xlm_base_cased.cfg
```

### XNLI avec Installation de Dépendances
```bash
bash ./flue/evaluation_auto.sh xnli-XLM true xnli_config_xlm_base_cased.cfg
```

### Tâche Hugging Face
```bash
bash ./flue/evaluation_auto.sh cls-books-HF false cls_books_lr5e6_hf.cfg
```

### Tâche PAWSX
```bash
bash ./flue/evaluation_auto.sh pawsx-HF true pawsx_config_hf.cfg
```

## Configuration

### Fichiers de Configuration Disponibles

Le framework inclut des configurations dans `flue/examples/` :
- `cls_books_lr5e6_xlm_base_cased.cfg` - Classification livres (XLM)
- `cls_books_lr5e6_hf.cfg` - Classification livres (HF)
- Autres configurations spécifiques aux tâches

### Structure des Fichiers de Configuration

```bash
# Paramètres du modèle
model_type=flaubert
model_name=flaubert_base_cased
model_name_or_path=flue/pretrained_models/flaubert_base_cased

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
```

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
"ma_nouvelle_tache")
    echo "Exécution de ma nouvelle tâche..."
    # Votre code d'implémentation ici
    ;;
```

2. **Ajouter à l'aide** :
```bash
# Dans la fonction show_usage(), ajouter :
echo "  ma_nouvelle_tache    : Description de ma tâche"
```

3. **Tester l'implémentation** :
```bash
bash ./flue/evaluation_auto.sh ma_nouvelle_tache false test_config.cfg
```

### Guidelines de Contribution

- Gardez la cohérence avec les tâches existantes
- Ajoutez des messages d'erreur appropriés
- Documentez les nouvelles tâches dans les README
- Testez avec différentes configurations

### Structure des Fichiers

- `evaluation_auto.sh` : Script principal d'évaluation
- `flue/examples/` : Fichiers de configuration
- `flue/data/` : Données d'évaluation
- Documentation dans les fichiers README
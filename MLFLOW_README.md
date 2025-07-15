# MLflow Integration for FLUE Benchmark

Ce document décrit l'intégration de MLflow dans le benchmark FLUE pour le suivi automatique des expériences.

## 🚀 Fonctionnalités

L'intégration MLflow ajoute les fonctionnalités suivantes au benchmark FLUE :

- **Suivi automatique des expériences** : Tous les hyperparamètres, métriques et artéfacts sont enregistrés automatiquement
- **Comparaison d'expériences** : Interface web pour comparer les performances de différents modèles
- **Reproductibilité** : Sauvegarde complète de la configuration et des résultats
- **Versioning des modèles** : Gestion des versions des modèles entraînés

## 📦 Installation

MLflow est automatiquement installé lors de l'exécution avec `INSTALL_LIBS=true` :

```bash
./flue/evaluation_auto.sh cls-books-HF true [modèle] [config]
```

Ou installation manuelle :
```bash
pip install mlflow
```

## 🎯 Utilisation

### Lancement d'une expérience avec MLflow

La tâche `cls-books-HF` a été modifiée pour inclure le suivi MLflow automatiquement :

```bash
# Exemple d'utilisation
./flue/evaluation_auto.sh cls-books-HF true flaubert_base_cased

# Avec une configuration personnalisée
./flue/evaluation_auto.sh cls-books-HF true camembert_base cls_books_custom.cfg
```

### Visualisation des résultats

#### Option 1 : Script automatique
```bash
./start_mlflow_ui.sh
```

#### Option 2 : Commande manuelle
```bash
mlflow ui --backend-store-uri ./mlflow_logs --host 0.0.0.0 --port 5000
```

Puis ouvrez votre navigateur sur : http://localhost:5000

## 📊 Métriques suivies

### Hyperparamètres enregistrés :
- `model_name` : Nom du modèle utilisé
- `learning_rate` : Taux d'apprentissage
- `num_epochs` : Nombre d'époques d'entraînement
- `batch_size` : Taille du batch
- `max_seq_length` : Longueur maximale des séquences
- `task` : Nom de la tâche (ex: cls-books)

### Métriques enregistrées :
- `validation_eval_accuracy` : Précision sur le jeu de validation
- `validation_eval_loss` : Perte sur le jeu de validation
- `test_accuracy` : Précision finale sur le jeu de test
- `train_loss` : Perte d'entraînement par époque
- `eval_loss` : Perte de validation par époque

### Artéfacts sauvegardés :
- Modèle complet entraîné
- Configuration du modèle (`config.json`)
- Tokenizer et vocabulaire
- Résultats d'évaluation (`eval_results.json`)
- Prédictions de test (`predict_results_None.txt`)
- Arguments d'entraînement

## 🗂️ Structure des dossiers

```
FLUE/
├── mlflow_logs/           # Base de données MLflow
├── flue/
│   ├── train_with_mlflow.py    # Script wrapper MLflow
│   └── evaluation_auto.sh      # Script principal modifié
├── start_mlflow_ui.sh          # Script pour lancer l'interface
└── MLFLOW_README.md           # Cette documentation
```

## 🔍 Interface MLflow

L'interface MLflow permet de :

1. **Comparer les expériences** : Tableau comparatif avec métriques et hyperparamètres
2. **Visualiser les métriques** : Graphiques d'évolution des métriques
3. **Télécharger les modèles** : Accès direct aux modèles entraînés
4. **Reproduire les expériences** : Informations complètes pour la reproductibilité

## 🛠️ Personnalisation

### Modifier l'emplacement de stockage

Par défaut, les logs MLflow sont stockés dans `./mlflow_logs`. Pour changer cela, modifiez le fichier `flue/train_with_mlflow.py` :

```python
# Ligne 30
mlflow.set_tracking_uri("file:///path/to/your/mlflow/logs")
```

### Ajouter des métriques personnalisées

Vous pouvez ajouter des métriques supplémentaires dans `flue/train_with_mlflow.py` :

```python
# Dans la fonction log_metrics_and_artifacts()
mlflow.log_metric("custom_metric", your_value)
```

## 🐛 Dépannage

### Problème : MLflow UI ne se lance pas
```bash
# Vérifier l'installation
pip install mlflow

# Vérifier les permissions
ls -la mlflow_logs/
```

### Problème : Expériences non visibles
```bash
# Vérifier le dossier MLflow
ls -la mlflow_logs/
# Le dossier doit contenir un fichier 'mlruns.db' ou un dossier 'mlruns/'
```

### Problème : Erreur de permission
```bash
# Sur Linux/WSL, donner les permissions
chmod -R 755 mlflow_logs/
```

## 📝 Exemples d'utilisation

### Comparaison de modèles
```bash
# Tester différents modèles
./flue/evaluation_auto.sh cls-books-HF true flaubert_base_cased
./flue/evaluation_auto.sh cls-books-HF true camembert_base
./flue/evaluation_auto.sh cls-books-HF true distilbert-base-multilingual-cased

# Comparer dans MLflow UI
./start_mlflow_ui.sh
```

### Tuning d'hyperparamètres
```bash
# Créer différentes configurations et les tester
cp flue/examples/cls_books_lr5e6_hf_base_uncased.cfg flue/examples/cls_books_lr1e5.cfg
# Modifier le learning rate dans le nouveau fichier
./flue/evaluation_auto.sh cls-books-HF true flaubert_base_cased cls_books_lr1e5.cfg
```

## 🚀 Prochaines étapes

L'intégration MLflow peut être étendue aux autres tâches FLUE :
- `xnli-HF` : Classification d'inférence textuelle
- `wsd` : Désambiguïsation lexicale
- `pawsx` : Paraphrase adverse

Contactez-nous pour l'extension à d'autres tâches !

---
**Auteur** : Aurélien Tassaux  
**Date** : Juillet 2025  
**Version** : 1.0

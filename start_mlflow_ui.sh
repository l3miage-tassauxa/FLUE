#!/usr/bin/env bash
# Script pour lancer l'interface MLflow UI
# Aurélien Tassaux

echo "=== Lancement de MLflow UI ==="

# Activer l'environnement conda si disponible
if command -v conda &> /dev/null; then
    echo "Activation de l'environnement hf-finetune..."
    eval "$(conda shell.bash hook)"
    conda activate hf-finetune
    
    if [ $? -eq 0 ]; then
        echo "✅ Environnement hf-finetune activé"
    else
        echo "⚠️  Impossible d'activer hf-finetune, utilisation de l'environnement par défaut"
    fi
fi

# Vérifier si MLflow est installé
if ! command -v mlflow &> /dev/null; then
    echo "❌ MLflow n'est pas installé ou n'est pas accessible."
    echo "Pour installer MLflow, exécutez: ./install_mlflow.sh"
    exit 1
fi

# Créer le dossier mlflow_logs s'il n'existe pas
if [ ! -d "./mlflow_logs" ]; then
    echo "Dossier mlflow_logs non trouvé. Création du dossier..."
    mkdir -p ./mlflow_logs
fi

# Vérifier le dossier courant
if [ "$(basename "$PWD")" != "FLUE" ]; then
    echo "Veuillez positionner le terminal dans le dossier FLUE, racine du git."
    exit 1
fi

echo "Démarrage de l'interface MLflow..."
echo "L'interface sera disponible sur:"
echo "  - http://localhost:5000 (depuis Windows)"
echo "  - http://127.0.0.1:5000 (alternative)"
echo "Appuyez sur Ctrl+C pour arrêter l'interface."
echo ""

# Lancer MLflow UI
mlflow ui --backend-store-uri ./mlflow_logs --host 127.0.0.1 --port 5000

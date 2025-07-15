#!/usr/bin/env bash
# Script pour lancer l'interface MLflow UI
# Aurélien Tassaux

echo "=== Lancement de MLflow UI ==="

# Vérifier si MLflow est installé
if ! command -v mlflow &> /dev/null; then
    echo "MLflow n'est pas installé. Veuillez l'installer avec: pip install mlflow"
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
echo "L'interface sera disponible sur: http://localhost:5000"
echo "Appuyez sur Ctrl+C pour arrêter l'interface."
echo ""

# Lancer MLflow UI
mlflow ui --backend-store-uri ./mlflow_logs --host 0.0.0.0 --port 5000

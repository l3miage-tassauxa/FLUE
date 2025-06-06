#!/usr/bin/env bash
# Aurélien Tassaux

# Check if the first argument is provided
if [ -z "$1" ]; then
        echo "Veuiller spécifier une tache."
        exit 1
fi
# Check if the second argument is provided
if [ -z "$2" ]; then
        echo "Veuiller spécifier si les librairies doivent être installées (true/false)."
        exit 1
fi

# Install required libraries
if [ $2 == true ]; then
    ./install_libraries.sh
    echo "Libraries installed."
else
    echo "Skipping library installation."
fi

# Lance les scripts de préparation des données et d'avaluation en fonction de la tâche spécifiée
case $1 in
    cls)
    echo "Getting CLS data..."
        ./get-data-cls.sh ./Data
        echo "Preparing CLS data..."
        ./flue/prepare-data-cls.sh ./Data ./Model/model.pth false
        ;;

    pawsx)
        echo "Getting PAWSX data..."
        ./get-data-xnli.sh ./Data
        echo "Preparing PAWSX data..."
        ./flue/prepare-data-pawsx.sh ./Data ./Model/model.pth false
        ;;

    xnli)
        echo "Getting XNLI data..."
        ./get-data-xnli.sh ./Data
        echo "Preparing XNLI data..."
        ./flue/prepare-data-xnli.sh ./Data ./Model/model.pth false
        ;;
    *)
        echo "Veuiller spécifier une tache valide."
        echo "Tâches valides: cls, pawsx, xnli"
        exit 1
        ;;
esac

#!/bin/bash
# manage_main.sh

# Étape 1 : Arrêter le processus main.py s'il est en cours d'exécution
echo "Vérification et arrêt de main.py s'il est en cours d'exécution..."
PID=$(pgrep -f main.py)
if [ -n "$PID" ]; then
    echo "Arrêt de main.py (PID: $PID)..."
    kill $PID
    sleep 2  # Attendre que le processus soit bien arrêté
else
    echo "main.py n'est pas en cours d'exécution."
fi

# Étape 2 : Faire un git pull pour mettre à jour le code
echo "Mise à jour du dépôt avec git pull..."
cd /chemin/vers/le/projet || exit
git pull

# Étape 3 : Relancer main.py en arrière-plan
echo "Relancer main.py en arrière-plan..."
nohup python main.py > output.log 2>&1 &
echo "main.py est relancé."

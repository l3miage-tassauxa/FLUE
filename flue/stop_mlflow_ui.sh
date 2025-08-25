#!/bin/bash
# Script to stop MLflow UI
# This script stops any running MLflow server processes

echo "Stopping MLflow UI for user $(whoami)..."

# Check if any MLflow processes are running for this user
USER=$(whoami)
if ps aux | grep -v grep | grep "$USER.*mlflow.server" > /dev/null; then
    echo "Found running MLflow processes for user $USER:"
    ps aux | grep -v grep | grep "$USER.*mlflow.server" | while read line; do
        echo "  $line"
    done
    echo "Stopping them..."
    
    # Get PIDs of MLflow processes owned by this user
    MLFLOW_PIDS=$(ps aux | grep -v grep | grep "$USER.*mlflow.server" | awk '{print $2}')
    
    if [ ! -z "$MLFLOW_PIDS" ]; then
        echo "Stopping MLflow processes with PIDs: $MLFLOW_PIDS"
        for pid in $MLFLOW_PIDS; do
            kill "$pid"
        done
        sleep 2
        
        # Verify they're stopped
        REMAINING_PIDS=$(ps aux | grep -v grep | grep "$USER.*mlflow.server" | awk '{print $2}')
        if [ ! -z "$REMAINING_PIDS" ]; then
            echo "Some processes didn't stop gracefully. Force killing PIDs: $REMAINING_PIDS"
            for pid in $REMAINING_PIDS; do
                kill -9 "$pid"
            done
            sleep 1
        fi
    fi
    
    echo "[OK] MLflow UI stopped successfully for user $USER."
else
    echo "No MLflow processes found running for user $USER."
fi

# Check if port 5000 is still in use by this user's processes
USER=$(whoami)
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "[WARNING]  Port 5000 is still in use. Checking if it's owned by user $USER..."
    PORT_OWNER=$(lsof -Pi :5000 -sTCP:LISTEN | tail -n +2 | awk '{print $3}' | head -1)
    if [ "$PORT_OWNER" = "$USER" ]; then
        echo "[WARNING]  Port 5000 is in use by your process:"
        lsof -Pi :5000 -sTCP:LISTEN | grep "$USER"
    else
        echo "[OK] Port 5000 is in use by another user ($PORT_OWNER), not interfering."
    fi
else
    echo "[OK] Port 5000 is now available."
fi

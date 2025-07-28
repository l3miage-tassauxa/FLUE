#!/bin/bash
# Script to stop MLflow UI
# This script stops any running MLflow server processes

echo "Stopping MLflow UI..."

# Check if any MLflow processes are running
if ps aux | grep -q "mlflow.server"; then
    echo "Found running MLflow processes. Stopping them..."
    pkill -f "mlflow.server"
    sleep 2
    
    # Verify they're stopped
    if ps aux | grep -q "mlflow.server"; then
        echo "Some processes didn't stop gracefully. Force killing..."
        pkill -9 -f "mlflow.server"
        sleep 1
    fi
    
    echo "✅ MLflow UI stopped successfully."
else
    echo "No MLflow processes found running."
fi

# Check if port 5000 is still in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5000 is still in use by another process:"
    lsof -Pi :5000 -sTCP:LISTEN
else
    echo "✅ Port 5000 is now available."
fi

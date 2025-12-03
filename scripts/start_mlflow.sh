#!/bin/bash

echo "🚀 Starting MLflow UI..."

# MLflow tracking URI
TRACKING_URI="outputs/mlruns"

# Create directory if not exists
mkdir -p $TRACKING_URI

# Check if port 5000 is in use
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 5000 is already in use"
    echo "   Trying to kill existing process..."
    lsof -ti:5000 | xargs kill -9
    sleep 2
fi

# Start MLflow UI
echo "📊 MLflow UI starting at http://localhost:5000"
echo "   Tracking URI: $TRACKING_URI"
echo ""
echo "Press Ctrl+C to stop"
echo ""

mlflow ui \
    --backend-store-uri $TRACKING_URI \
    --host 0.0.0.0 \
    --port 5000
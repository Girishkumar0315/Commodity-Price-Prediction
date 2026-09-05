#!/bin/bash
# AgriML — Commodity Price Prediction Startup Script

echo "============================================"
echo "  AgriML — Commodity Price Prediction"
echo "============================================"

# Move to project root
cd "$(dirname "$0")"

# Install dependencies if needed
echo "[1/3] Checking dependencies..."
pip install -r requirements.txt -q --break-system-packages

# Train model and generate charts
echo "[2/3] Training models & generating charts..."
python train_model.py

# Start Flask server
echo "[3/3] Starting web server..."
echo ""
echo "  ✅ Open in browser → http://localhost:5000"
echo ""
python app.py

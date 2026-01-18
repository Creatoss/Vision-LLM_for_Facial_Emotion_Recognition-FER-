#!/bin/bash
# ============================================================
# Streamlit Emotion Recognition Dashboard - Startup Script
# Unix/Linux/macOS Bash Script
# ============================================================

echo ""
echo "========================================"
echo "  Emotion Recognition Dashboard"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "[1/4] Checking Python version..."
python3 --version

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "[2/4] Virtual environment found, activating..."
    source venv/bin/activate
else
    echo "[2/4] Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
fi

# Check if requirements are installed
echo "[3/4] Checking dependencies..."
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements_streamlit.txt
fi

# Check if model exists
echo "[4/4] Checking model files..."
if [ ! -d "blip2-emotion-rafce-final" ]; then
    echo ""
    echo "Warning: Fine-tuned model not found!"
    echo "Please download from Google Drive:"
    echo "  /content/drive/MyDrive/blip2-emotion-rafce-final"
    echo "And extract to: blip2-emotion-rafce-final/"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Launch dashboard
echo ""
echo "Starting Streamlit dashboard..."
echo ""
echo "Dashboard will open at: http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

python -m streamlit run streamlit_app.py

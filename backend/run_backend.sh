#!/bin/bash

echo "🚀 Starting FastAPI Backend..."

# Activate virtual environment if exists
if [ -d "../venv_mediapipe" ]; then
    source ../venv_mediapipe/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

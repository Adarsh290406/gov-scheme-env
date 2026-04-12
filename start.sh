#!/bin/bash
# Start the FastAPI server in background
uvicorn app:app --host 0.0.0.0 --port 7860 &

# Run inference.py in foreground (evaluator reads its stdout)
python inference.py

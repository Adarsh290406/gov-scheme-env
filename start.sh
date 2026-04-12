#!/bin/bash
uvicorn app:app --host 0.0.0.0 --port 7860 2>/dev/null &
sleep 5
python inference.py
wait

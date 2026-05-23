#!/bin/sh
# Start the FastAPI backend (uvicorn) and then the Gradio UI.
# The Gradio UI runs in the foreground so the container stays alive.

API_PORT=${API_PORT:-8080}
UI_PORT=${PORT:-7860}

echo "Starting API on port ${API_PORT}..."
nohup uvicorn app:app --host 0.0.0.0 --port ${API_PORT} --log-level info > /proc/1/fd/1 2>/proc/1/fd/2 &

echo "Starting Gradio UI on port ${UI_PORT}..."
exec python gradio_app.py

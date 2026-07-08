#!/bin/bash
# Starts FastAPI privately on 8000, then Streamlit on Cloud Run's public $PORT.
# Streamlit's requests to the backend use API_BASE_URL=http://127.0.0.1:8000 (internal only).

set -e

echo "Starting FastAPI backend on internal port 8000..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

# Give the backend a moment to come up before Streamlit starts hitting it
sleep 3

echo "Starting Streamlit frontend on public port ${PORT}..."
export API_BASE_URL="http://127.0.0.1:8000"
streamlit run frontend/app.py \
    --server.port "${PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
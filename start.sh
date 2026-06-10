#!/bin/bash
set -e

echo "=== Dark Store Forecast App ==="
echo "Starting FastAPI on port ${PORT_API:-8000}..."
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT_API:-8000}" \
    --workers 1 \
    --log-level warning &

sleep 4

echo "Starting Streamlit on port ${PORT_APP:-7860}..."
streamlit run streamlit_app/app.py \
    --server.port "${PORT_APP:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.maxUploadSize 50 \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
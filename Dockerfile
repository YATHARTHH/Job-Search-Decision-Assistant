# Step 8 - Dockerfile: runs FastAPI (backend) + Streamlit (frontend) in one container.
# Cloud Run only exposes ONE port per service, so Streamlit (the user-facing UI) listens
# on the port Cloud Run gives us ($PORT), and FastAPI runs privately on localhost:8000
# inside the same container - Streamlit calls it internally, never exposed externally.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY outputs/ ./outputs/
COPY start.sh .
RUN chmod +x start.sh

# Cloud Run sets $PORT at runtime (usually 8080) - Streamlit must listen on it
ENV PORT=8080
EXPOSE 8080

CMD ["./start.sh"]
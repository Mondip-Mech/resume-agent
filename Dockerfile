FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
# build-essential: needed by chromadb and sentence-transformers native extensions
# libglib2.0-0: required by PyMuPDF (headless PDF parsing — no OpenGL needed)
# curl: used by HEALTHCHECK
# git: required by sentence-transformers to fetch model metadata
# Note: libgl1-mesa-glx was removed in Debian Trixie; PyMuPDF text extraction
#       does not require any OpenGL package at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download sentence-transformers model ───────────────────────────────────
# Bakes the 80 MB model into the image so runtime startup is instant.
# The model is cached at /root/.cache/huggingface/hub inside this layer.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading all-MiniLM-L6-v2...'); \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model cached successfully.')"

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# Runtime directories
RUN mkdir -p uploads output chroma_db

# ── Ports ─────────────────────────────────────────────────────────────────────
# 8000 = FastAPI REST API (default CMD)
# 8501 = Streamlit UI (override CMD in docker-compose)
EXPOSE 8000
EXPOSE 8501

# ── Health check (for the API service) ────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# ── Default: start FastAPI ─────────────────────────────────────────────────────
# To run Streamlit instead, override CMD in docker-compose.yml:
#   command: streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

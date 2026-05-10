# Hugging Face Spaces deployment image for FinIntel.
# Per HF's deprecation of the Streamlit SDK, all Streamlit Spaces are now
# Docker-based. This Dockerfile is the runtime contract HF builds against.

FROM python:3.12-slim

# Cache HF models inside the image at a persistent path. Do NOT use /tmp
# on HF Spaces — /tmp is treated as ephemeral, so build-time downloads
# disappear at container start and have to be re-fetched on first query.
# That re-fetch (~440 MB for BGE) takes longer than HF's reverse proxy
# tolerates on the SSE/WebSocket connection, which is what causes the
# Streamlit page to go blank with no error.
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first so the layer caches when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Pre-download the BGE embedding model into the image. This bakes ~440 MB
# of model weights into a layer so the first user query loads from disk
# in <2s instead of triggering a download. Build time goes up by ~2 min
# but every cold start after that is fast.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Copy the rest of the application code.
COPY . .

# HF Streamlit Spaces require port 8501.
EXPOSE 8501

# --server.address=0.0.0.0      : reachable from outside the container
# --server.headless=true        : don't try to open a browser window
# --server.fileWatcherType=none : silence the transformers introspection noise
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]

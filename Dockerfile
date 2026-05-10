# Hugging Face Spaces deployment image for FinIntel.
# Per HF's deprecation of the Streamlit SDK, all Streamlit Spaces are now
# Docker-based. This Dockerfile is the runtime contract HF builds against.

FROM python:3.12-slim

# Redirect ML model caches to /tmp (HF Spaces home dir is occasionally
# read-only; /tmp is always writable for the runtime user).
ENV HF_HOME=/tmp/huggingface \
    TRANSFORMERS_CACHE=/tmp/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first so the layer caches when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

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

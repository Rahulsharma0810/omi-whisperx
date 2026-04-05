FROM python:3.12-slim

ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch separately so we can pick CPU-only wheels on arm64 (RPi5).
# On amd64 the standard index is fine and pulls CUDA-capable wheels.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
      pip install --no-cache-dir torch torchaudio \
        --index-url https://download.pytorch.org/whl/cpu; \
    else \
      pip install --no-cache-dir torch torchaudio; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py benchmark.py ui.html benchmark.html ./

# HuggingFace model cache and speaker profiles — mount a volume here
ENV HF_HOME=/data/huggingface
ENV PROFILES_DIR=/data/speakers

# Suppress noisy pyannote/torchcodec warnings about missing CUDA libs on CPU containers
ENV PYTHONWARNINGS="ignore::UserWarning:pyannote"

# Server defaults (override with -e in Portainer / docker run)
ENV WHISPER_MODEL=medium
ENV WHISPER_BATCH_SIZE=16
ENV SPEAKER_THRESHOLD=0.80

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]

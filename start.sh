#!/usr/bin/env zsh
set -e

SCRIPT_DIR="${0:A:h}"
VENV="$HOME/.venvs/whisperx"

# First-time setup
if [[ ! -d "$VENV" ]]; then
  echo "Creating venv at $VENV ..."
  python3.12 -m venv "$VENV"
  source "$VENV/bin/activate"
  pip install -r "$SCRIPT_DIR/requirements.txt"
else
  source "$VENV/bin/activate"
fi

export WHISPER_MODEL="${WHISPER_MODEL:-medium}"
export WHISPER_BATCH_SIZE="${WHISPER_BATCH_SIZE:-16}"

# Content filter — Ollama LLM classifier
# Set OLLAMA_ENABLED=true and point OLLAMA_URL at the machine running Ollama
# export OLLAMA_ENABLED="true"
# export OLLAMA_URL="http://192.168.1.X:11434"   # replace with your Ollama machine IP
# export OLLAMA_MODEL="gemma2:2b"                 # or llama3.2:3b, mistral, etc.

echo "Starting omi-whisperx server (model=$WHISPER_MODEL, ollama=${OLLAMA_ENABLED:-false}) ..."
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --app-dir "$SCRIPT_DIR" \
  --log-level info

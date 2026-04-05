#!/usr/bin/env zsh
set -e

SCRIPT_DIR="${0:A:h}"
VENV="$HOME/.venvs/whisperx"

# Load .env if present (does not override vars already set in the environment)
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -o allexport
  source "$SCRIPT_DIR/.env"
  set +o allexport
fi

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

# Content filter — cascade: NLI (fast) → Ollama (fallback when NLI uncertain)
# export CONTENT_FILTER="true"             # set false to disable entirely

# Tier 1: Zero-shot NLI (runs locally, no extra server needed)
# export NLI_ENABLED="true"               # default: true
# export NLI_MODEL="typeform/distilbert-base-uncased-mnli"
# export NLI_THRESHOLD="0.85"             # confidence cutoff; below this → Ollama

# Tier 2: Ollama (only called when NLI is uncertain)
# export OLLAMA_ENABLED="true"
# export OLLAMA_URL="http://192.168.1.X:11434"  # your Ollama machine IP
# export OLLAMA_MODEL="gemma2:2b"               # or llama3.2:3b, mistral:7b

echo "Starting omi-whisperx server (model=$WHISPER_MODEL, ollama=${OLLAMA_ENABLED:-false}) ..."
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --app-dir "$SCRIPT_DIR" \
  --log-level info

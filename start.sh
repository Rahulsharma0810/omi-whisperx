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
export MIN_SECONDS="${MIN_SECONDS:-8}"

echo "Starting omi-whisperx server (model=$WHISPER_MODEL, min_seconds=$MIN_SECONDS) ..."
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --app-dir "$SCRIPT_DIR" \
  --log-level info

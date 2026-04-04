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
# SPEAKER_THRESHOLD default is 0.80 (set in server.py); override here if needed
# export SPEAKER_THRESHOLD="0.80"
# WHISPER_INITIAL_PROMPT: default Hindi/English bilingual prompt is set in server.py

echo "Starting omi-whisperx server (model=$WHISPER_MODEL, batch=$WHISPER_BATCH_SIZE) ..."
exec uvicorn server:app \
  --host 0.0.0.0 \
  --port 8080 \
  --app-dir "$SCRIPT_DIR" \
  --log-level info

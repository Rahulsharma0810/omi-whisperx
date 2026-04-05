# Agents Guide — omi-whisperx

Guidance for AI coding agents working on this repository.

## Project Overview

FastAPI service that transcribes audio using WhisperX, diarizes speakers, and resolves speaker names
via voice embeddings. Designed to run as a persistent background server connected to an Omi wearable
device via a Cloudflare tunnel.

## Key Files

| File | Purpose |
|---|---|
| `server.py` | Main FastAPI app — transcription, diarization, speaker logic, content filter, SSE, API routes |
| `ui.html` | Live monitoring dashboard — served at `/ui`, receives pipeline events via SSE |
| `benchmark.py` | Standalone pipeline benchmark — times each stage, outputs RTF, supports `--compare` |
| `start.sh` | Launches the server — sets env vars, activates venv, starts uvicorn on port 8080 |
| `requirements.txt` | Python dependencies — do not add new packages without a strong reason |
| `cloudflared-config.yml` | Cloudflare tunnel config — exposes port 8080 to the public URL |

## Architecture

```
Omi device → cloudflared tunnel → uvicorn (port 8080) → FastAPI
                                                          ├── /inference  (transcribe + diarize)
                                                          ├── /speakers   (list/delete profiles)
                                                          ├── /filter     (content filter status)
                                                          ├── /health     (model info)
                                                          ├── /ui         (live dashboard HTML)
                                                          └── /events     (SSE stream for dashboard)
```

Models are loaded **once at startup** (not per request):
- `whisperx.load_model` — transcription (CTranslate2, CPU+int8 or CUDA+float16)
- `DiarizationPipeline` — speaker diarization (MPS on Apple Silicon, CPU on RPi5)
- `VoiceEncoder` — speaker embeddings (always CPU — resemblyzer has no MPS support)
- `transformers.pipeline("zero-shot-classification")` — NLI content filter (MPS/CUDA/CPU), only loaded when `CONTENT_FILTER=true` and `NLI_ENABLED=true`

`load_align_model` is called **per request** because alignment model choice depends on detected language.

## Device Rules (do not change)

```python
# CTranslate2 (whisperx backend) does NOT support MPS — CPU or CUDA only
WHISPER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE   = "float16" if CUDA else "int8"

# PyTorch models (alignment, diarization) support MPS on Apple Silicon
TORCH_DEVICE = "mps" | "cuda" | "cpu"  # in priority order

# resemblyzer has no MPS support — always CPU
VoiceEncoder(device="cpu")
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Model size: tiny/base/small/medium/large-v2 |
| `WHISPER_BATCH_SIZE` | `16` | Transcription batch size |
| `WHISPER_INITIAL_PROMPT` | Hindi/English bilingual prompt | Primes Whisper for code-switching |
| `HF_TOKEN` | — | HuggingFace token for diarization model download |
| `PROFILES_DIR` | `~/.omi/speakers` | Speaker embedding storage directory |
| `SPEAKER_THRESHOLD` | `0.80` | Cosine similarity threshold for speaker name matching |
| `CONTENT_FILTER` | `true` | Set `false` to disable the entire content filter |
| `NLI_ENABLED` | `true` | Enable zero-shot NLI classifier (Tier 1) |
| `NLI_MODEL` | `typeform/distilbert-base-uncased-mnli` | HuggingFace zero-shot model |
| `NLI_THRESHOLD` | `0.85` | Confidence cutoff — below this NLI escalates to Ollama |
| `OLLAMA_ENABLED` | `false` | Enable Ollama fallback classifier (Tier 2) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `gemma2:2b` | Ollama model name |
| `OLLAMA_TIMEOUT` | `5` | Seconds before Ollama call is abandoned (returns keep) |

## Running Locally

```bash
# Server
HF_TOKEN=your_token ./start.sh

# Benchmark (Mac M2)
python benchmark.py audio.wav --trials 3 --output json --output-file results_mac.json

# Benchmark (Raspberry Pi 5)
python benchmark.py audio.wav --trials 3 --batch-size 1 --output json --output-file results_rpi5.json

# Compare results
python benchmark.py --compare results_mac.json results_rpi5.json

# Quick benchmark (no diarization needed, works without HF_TOKEN)
python benchmark.py --no-diarization --no-embedding --language en --trials 1
```

## Raspberry Pi 5 Setup

```bash
sudo apt-get install -y python3.12 python3.12-venv ffmpeg libsndfile1
python3.12 -m venv ~/.venvs/whisperx
source ~/.venvs/whisperx/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

RPi5 notes:
- `TORCH_DEVICE` will be `cpu` automatically (no MPS/CUDA)
- Use `--batch-size 1` to reduce peak RAM
- Diarization on CPU: expect ~60–120s for 30s audio vs ~18s on M2 MPS
- Set `HF_HOME` to a USB SSD path if using an SD card boot drive

## Inference Pipeline (server.py)

```
POST /inference
  0. Emit SSE            emit_event("audio_received")          → live dashboard
  1. Transcribe          model.transcribe(audio_path)         → segments + detected_lang
     Emit SSE            emit_event("transcribed")
  2. Align               load_align_model() + whisperx.align() → word-level timestamps
  3. Diarize             diarize_model(audio_path)             → SPEAKER_00, SPEAKER_01 ...
  4. Assign speakers     assign_word_speakers()                → words labelled by speaker
  5. Embed               voice_encoder.embed_utterance()       → per-speaker numpy vectors
  6. Resolve names       cosine similarity vs stored profiles  → "Rahul" instead of SPEAKER_00
  7. Content filter      classify_content()                    → drop entertainment audio
     Emit SSE            emit_event("nli_result" | "ollama_result" | "filtered" | "transcript")
  8. Return              JSON with segments, text, duration
```

## Content Filter

Cascade classifier — classifies transcripts before Omi creates a memory.
Returns `{"segments": [], "text": ""}` for entertainment so Omi saves nothing.

**Tier 1 — Zero-shot NLI** (`NLI_ENABLED=true`, default on):
- Runs inside the Python process via HuggingFace `zero-shot-classification` pipeline
- Uses MPS on Apple Silicon, CUDA on Nvidia, CPU fallback
- Labels: `"gaming, sitcom, movie, TV show, or scripted fiction"` vs `"tutorial, educational video..."`
- If confidence ≥ `NLI_THRESHOLD` (default 0.85) → result is final, Ollama never called
- If confidence < threshold → escalates to Tier 2

**Tier 2 — Ollama** (`OLLAMA_ENABLED=false` by default):
- HTTP POST to Ollama `/api/generate` with a one-shot classification prompt
- Only called when NLI is uncertain (e.g. Hinglish, ambiguous content)
- Times out after `OLLAMA_TIMEOUT` seconds → returns `keep` (safe default)
- Point at a remote machine: `OLLAMA_URL=http://192.168.1.X:11434`

**Safe defaults:** on any failure (NLI error, Ollama unreachable, unclear answer) the classifier returns `keep` — never drops content silently.

## Live Dashboard (`/ui`)

`GET /ui` — serves `ui.html`, a self-contained HTML/JS dashboard.
`GET /events` — Server-Sent Events stream; one `asyncio.Queue` per connected client.

`emit_event(type, data)` is a sync helper called from the async inference route and from
`classify_content()`. It puts JSON onto every client queue. Full queues (disconnected clients)
are pruned automatically.

Events emitted per chunk:

| Event | When | Key fields |
|---|---|---|
| `audio_received` | Start of inference | `chunk_id`, `size_kb` |
| `transcribed` | After whisperx | `chunk_id`, `lang`, `segments` |
| `nli_result` | After NLI runs | `chunk_id`, `decision`, `confidence`, `confident` |
| `ollama_result` | After Ollama runs | `chunk_id`, `decision` |
| `filtered` | Chunk dropped | `chunk_id`, `preview` |
| `transcript` | Chunk saved | `chunk_id`, `lang`, `duration`, `segments[]` |

## Omi App Configuration

**Public URL:** `https://baata.rvsharma.com` (Cloudflare tunnel → `localhost:8080`)
Use this URL in the Omi app — not the local IP — so the pendant works away from home Wi-Fi.

**Transcript Provider config:**
```json
{
  "url": "https://baata.rvsharma.com/inference",
  "request_type": "multipart_form",
  "headers": null,
  "params": {
    "temperature": "0.0",
    "response_format": "verbose_json"
  },
  "audio_field_name": "file"
}
```

**Response Schema:**
```json
{
  "segments_path": "segments",
  "segments_text_field": "text",
  "segments_start_field": "start",
  "segments_end_field": "end",
  "segments_speaker_field": "speaker",
  "text_path": "text",
  "default_segment_duration": 5.0
}
```

Notes:
- `temperature_inc` is a Whisper.cpp param — not supported here, do not add it
- `language` param is optional — omit it to let Whisper auto-detect per chunk (best for Hinglish)
- Adding `"language": "hi"` forces Hindi mode; `"language": "en"` forces English

## Speaker Enrollment

Say the trigger phrase during a conversation:
> "remember/save/call/name this voice as NAME"

The next unrecognised speaker in that chunk is enrolled as NAME and saved to `PROFILES_DIR`.

## Docker / Portainer

The CI builds a multi-arch image (`linux/amd64` + `linux/arm64`) on every push to `main`
and pushes it to GHCR:

```
ghcr.io/rahulsharma0810/omi-whisperx:latest
```

**First-time setup:** After the first CI run, go to
`github.com/Rahulsharma0810` → Packages → `omi-whisperx` → Package settings → **Make public**
so Portainer can pull without credentials.

**Portainer stack (docker-compose):**

```yaml
version: "3.9"
services:
  omi-whisperx:
    image: ghcr.io/rahulsharma0810/omi-whisperx:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      HF_TOKEN: your_token_here
      WHISPER_MODEL: medium          # tiny/base/small/medium/large-v2
      WHISPER_BATCH_SIZE: "4"        # lower on RPi5 to save RAM
      SPEAKER_THRESHOLD: "0.80"
      CONTENT_FILTER: "true"
      NLI_ENABLED: "true"
      NLI_THRESHOLD: "0.85"
      OLLAMA_ENABLED: "false"        # set true + OLLAMA_URL to enable Tier 2
      OLLAMA_URL: "http://192.168.1.X:11434"
      OLLAMA_MODEL: "gemma2:2b"
    volumes:
      - omi_data:/data               # model cache + speaker profiles
volumes:
  omi_data:
```

**Key files:**
- `Dockerfile` — multi-arch build; installs CPU-only torch on arm64; copies `server.py`, `benchmark.py`, `ui.html`
- `.github/workflows/docker.yml` — triggers on changes to server.py, requirements.txt, or Dockerfile
- `/data/huggingface` — HuggingFace model cache (mount a volume — models are ~2–4 GB)
- `/data/speakers` — speaker embedding profiles (persistent)

**Build notes:**
- `build-essential` + `python3-dev` are installed in the image — required to compile `webrtcvad` (C extension pulled by whisperx)
- arm64 CI build via QEMU takes ~20–30 min — this is normal
- Models are **not** baked into the image; they download on first container start (~5 min)

Live dashboard is available at `http://<host>:8080/ui` once the container is running.

## Do Not

- Add MPS support to the whisperx transcription call — CTranslate2 does not support it
- Add MPS support to `VoiceEncoder` — resemblyzer does not support it
- Load `load_align_model` at startup — it must stay per-request (language varies per audio)
- Add new pip dependencies to `requirements.txt` without checking they have aarch64 wheels
  (the service must also run on Raspberry Pi 5)
- Add keyword lists back to the content filter — they were removed intentionally; NLI+Ollama handle classification
- Make `classify_content()` synchronous — it must stay `async` (Ollama call is async HTTP)
- Call `_classify_with_nli()` directly from async code without `asyncio.to_thread` — the HuggingFace pipeline is blocking

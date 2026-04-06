# omi-whisperx

A self-hosted, real-time audio transcription server for the [Omi](https://www.omi.me/) wearable device. Powered by [WhisperX](https://github.com/m-bain/whisperX), it transcribes conversations with speaker labels, identifies enrolled voices by name, and filters out entertainment/non-informative audio before Omi creates a memory.

Runs on **Mac (Apple Silicon)**, **Raspberry Pi 5**, or any **Linux/CUDA** machine.

---

## Features

- **Real-time transcription** via WhisperX (CTranslate2-optimized Whisper)
- **Speaker diarization** — who spoke, when
- **Speaker identification** — matches voices to saved profiles ("Rahul" instead of SPEAKER_00)
- **Voice enrollment** — say *"remember this voice as NAME"* to register a new speaker on the fly
- **Two-tier content filter** — drops entertainment audio (movies, TV, gaming) before Omi saves a memory
  - Tier 1: Fast local NLI classifier (zero-shot, runs on device)
  - Tier 2: Ollama LLM fallback for ambiguous cases
- **Live monitoring dashboard** at `/ui` — SSE stream shows the full pipeline in real time
- **Built-in benchmark tool** — measures RTF and per-stage latency; compare hardware profiles
- **Push notifications** via ntfy.sh for pipeline alerts
- **Multi-platform Docker image** — arm64 (Raspberry Pi) + amd64 (Linux/CUDA)

---

## Architecture

```
Omi wearable
     │  HTTP (audio chunks)
     ▼
Cloudflare Tunnel  ──►  omi-whisperx (port 8080)
                              │
                    ┌─────────┴──────────┐
                    │   FastAPI / uvicorn │
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        WhisperX         Diarization     VoiceEncoder
       (CTranslate2)    (pyannote-audio) (resemblyzer)
       CPU / CUDA        MPS / CUDA / CPU   CPU only
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Content Filter   │
                    │  NLI → Ollama (opt)│
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  JSON response to  │
                    │    Omi app         │
                    └────────────────────┘
```

**Inference pipeline per audio chunk:**

1. Receive audio → emit SSE `audio_received`
2. Transcribe with WhisperX → emit `transcribed`
3. Align to word-level timestamps
4. Diarize speakers (SPEAKER_00, SPEAKER_01 …)
5. Embed each speaker utterance (voice vector)
6. Resolve speaker names via cosine similarity against stored profiles
7. Classify content (NLI → Ollama) → emit `nli_result` / `ollama_result` / `filtered` / `transcript`
8. Return JSON with segments, full text, language, and duration

---

## Quick Start

### Prerequisites

- Python 3.12
- `ffmpeg` and `libsndfile1` (system packages)
- A [HuggingFace token](https://huggingface.co/settings/tokens) — required to download the diarization model ([pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)); you must also accept the model's license on HuggingFace

### 1. Clone & configure

```bash
git clone https://github.com/Rahulsharma0810/omi-whisperx
cd omi-whisperx
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
WHISPER_MODEL=medium   # or small / large-v2
HF_TOKEN=hf_...        # your HuggingFace token
```

### 2. Start the server

```bash
./start.sh
```

This creates a Python 3.12 venv at `~/.venvs/whisperx`, installs dependencies, and launches uvicorn on **port 8080**.

First start downloads WhisperX and diarization models (~2–4 GB). Subsequent starts are fast.

---

## Docker

### Pull and run

```bash
docker run -d \
  --name omi-whisperx \
  -p 8080:8080 \
  -e WHISPER_MODEL=medium \
  -e HF_TOKEN=hf_... \
  -v ~/.omi/huggingface:/data/huggingface \
  -v ~/.omi/speakers:/data/speakers \
  ghcr.io/rahulsharma0810/omi-whisperx:latest
```

### Docker Compose (Portainer or standalone)

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
      WHISPER_MODEL: medium          # tiny | base | small | medium | large-v2
      WHISPER_BATCH_SIZE: "4"        # use 4 on Raspberry Pi 5
      SPEAKER_THRESHOLD: "0.80"
      CONTENT_FILTER: "true"
      NLI_ENABLED: "true"
      NLI_THRESHOLD: "0.85"
      OLLAMA_ENABLED: "false"        # set true + OLLAMA_URL to enable Tier 2
      OLLAMA_URL: "http://192.168.1.X:11434"
      OLLAMA_MODEL: "gemma2:2b"
    volumes:
      - omi_data:/data
volumes:
  omi_data:
```

> **Note:** Models are not baked into the image. They download on first container start (~5 min).

---

## Raspberry Pi 5 Setup

```bash
# Install system dependencies
sudo apt-get install -y python3.12 python3.12-venv ffmpeg libsndfile1

# Create venv and install CPU-only torch first
python3.12 -m venv ~/.venvs/whisperx
source ~/.venvs/whisperx/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

RPi5 notes:
- Set `WHISPER_BATCH_SIZE=4` (default 16 is too high)
- Expect diarization ~60–120s for 30s of audio (vs ~18s on Mac M2)
- If booting from SD card, set `HF_HOME` to a USB SSD path for the model cache

---

## Omi App Configuration

In the Omi app, add a **Transcript Provider** pointing at your server (use a public URL — Cloudflare tunnel recommended):

| Field | Value |
|---|---|
| URL | `https://your-server.example.com/inference` |
| Request type | `multipart_form` |
| Audio field name | `file` |
| Params | `temperature=0.0`, `response_format=verbose_json` |

**Response schema mapping:**

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

Omit the `language` param to let Whisper auto-detect per chunk — best for multilingual / code-switching audio.

---

## Web UI

| URL | Description |
|---|---|
| `/ui` | Live monitor — real-time pipeline events as audio is processed |
| `/ui/speakers` | Speaker manager — listen to unidentified clips, assign names, enroll, rename, delete |
| `/benchmark` | Benchmark tool — upload/record/synthetic audio, per-stage RTF, side-by-side comparison |
| `/health` | Full system status — model, devices, speaker count, filter config, app version |

---

## API Reference

### `POST /inference`

Main transcription endpoint. Accepts a multipart audio file.

**Form fields:** `file` (audio), `language` (optional), `temperature` (default `0.0`), `response_format` (default `verbose_json`)

**Response:**

```json
{
  "segments": [
    { "start": 0.0, "end": 2.4, "text": "Hello world", "speaker": "Rahul" }
  ],
  "text": "Hello world",
  "language": "en",
  "duration": 5.2
}
```

When content is filtered (entertainment detected), returns `{"segments": [], "text": ""}` so Omi creates no memory.

---

### Speaker profiles

| Method | Path | Description |
|---|---|---|
| `GET` | `/speakers` | List enrolled speaker names and capture state |
| `POST` | `/speakers` | Enroll from audio — form fields: `name`, `file` (WAV/M4A/MP3) |
| `PATCH` | `/speakers/{name}` | Rename — form field: `new_name` |
| `DELETE` | `/speakers/{name}` | Delete one profile |
| `DELETE` | `/speakers` | Reset all profiles and capture state |

### Unidentified recordings

Up to 5 clips are saved per unique unknown voice. Visit `/ui/speakers` to listen and assign names.

| Method | Path | Description |
|---|---|---|
| `GET` | `/speakers/recordings` | List clips with best-match speaker suggestion and similarity score |
| `GET` | `/speakers/recordings/{id}/audio` | Stream WAV audio for playback |
| `POST` | `/speakers/recordings/{id}/assign` | Assign name → enroll speaker, delete clip. Form field: `name` |
| `POST` | `/speakers/recordings/purge` | Delete all clips that now match an enrolled speaker |
| `POST` | `/speakers/recordings/{id}/block` | Block voice permanently — saves to blocked list, deletes clip and all similar clips |
| `DELETE` | `/speakers/recordings/{id}` | Discard a clip without enrolling |
| `GET` | `/speakers/blocked` | Count of blocked voices |
| `DELETE` | `/speakers/blocked` | Clear all blocked voices |

---

### Live dashboard

| Method | Path | Description |
|---|---|---|
| `GET` | `/events` | SSE stream for `/ui` dashboard |

Events emitted per chunk:

| Event | When | Key fields |
|---|---|---|
| `audio_received` | Start of inference | `chunk_id`, `size_kb` |
| `transcribed` | After WhisperX | `chunk_id`, `lang`, `segments` |
| `nli_result` | After NLI | `chunk_id`, `decision`, `confidence`, `scores` |
| `ollama_result` | After Ollama | `chunk_id`, `decision`, `raw_response` |
| `filtered` | Chunk dropped | `chunk_id`, `preview` |
| `transcript` | Chunk saved | `chunk_id`, `lang`, `duration`, `segments[]` |

---

### Benchmark

| Method | Path | Description |
|---|---|---|
| `GET` | `/benchmark` | Benchmark UI |
| `GET` | `/benchmark/events` | SSE stream for benchmark progress |
| `POST` | `/benchmark/run` | Start benchmark — form fields: `trials`, `language`, `duration`, optional audio `file` |
| `GET` | `/benchmark/status` | `{"running": true/false}` |
| `POST` | `/benchmark/cancel` | Cancel a running benchmark |

---

### `GET /health`

Full system status: model, devices, speaker profiles, filter config, app version.

### `GET /filter`

Content filter configuration (enabled flags, NLI model, Ollama URL/model/timeout).

---

## Configuration Reference

All settings are environment variables. Copy `.env.example` to `.env` to get started.

### Whisper

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Model size: `tiny` `base` `small` `medium` `large-v2` |
| `WHISPER_BATCH_SIZE` | `16` | Transcription batch size (use `4` on RPi5) |
| `WHISPER_INITIAL_PROMPT` | bilingual | Override the Whisper initial prompt |

### HuggingFace

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | **Required.** Token for diarization model download |

### Speaker matching

| Variable | Default | Description |
|---|---|---|
| `PROFILES_DIR` | `~/.omi/speakers` | Directory for voice embedding profiles |
| `SPEAKER_THRESHOLD` | `0.80` | Cosine similarity cutoff for name resolution |

### Content filter

| Variable | Default | Description |
|---|---|---|
| `CONTENT_FILTER` | `true` | Set `false` to disable entirely |
| `NLI_ENABLED` | `true` | Enable zero-shot NLI classifier (Tier 1) |
| `NLI_MODEL` | `typeform/distilbert-base-uncased-mnli` | HuggingFace zero-shot model |
| `NLI_THRESHOLD` | `0.85` | Confidence cutoff; below this escalates to Ollama |
| `OLLAMA_ENABLED` | `false` | Enable Ollama fallback (Tier 2) |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `gemma2:2b` | Ollama model name |
| `OLLAMA_TIMEOUT` | `5` | Seconds before Ollama is abandoned (returns `keep`) |

### Notifications

| Variable | Default | Description |
|---|---|---|
| `NTFY_URL` | — | ntfy.sh topic URL for pipeline alerts |
| `NTFY_TOKEN` | — | ntfy.sh auth token |

---

## Speaker Enrollment

Three ways to enroll a speaker:

**1. Trigger phrase** — say during any conversation:
> *"remember this voice as NAME"* / *"save this voice as NAME"* / *"recognize my voice as NAME"*

The next unrecognised speaker in that chunk is enrolled as NAME.

**2. Upload or record at `/ui/speakers`** — upload any audio clip (WAV, M4A, MP3, voice note) or record directly from your browser mic. No cooperation from the other person required.

**3. Assign from recordings** — every time an unknown voice is heard, a short clip is saved automatically (max 5 per unique voice). Go to `/ui/speakers`, play each clip, and click **confirm** to assign a name. The similarity score shown next to each clip tells you how closely it matches enrolled speakers — use **confirm all suggestions** to bulk-assign in one click.

Multiple assignments to the same name average the embeddings together, building a more robust voiceprint over time.

**Blocking TV/movie voices** — when watching TV or a movie, unknown actor voices will appear in recordings. Click **block** on any clip to permanently ignore that voice. The embedding is saved to `~/.omi/blocked/` and that voice is never recorded again. Use **clear blocked** in the UI to undo all blocks. Unassigned clips auto-expire after `RECORDINGS_MAX_AGE_DAYS` days (default 7).

---

## Content Filter

The filter runs a two-tier cascade classifier on every transcript before Omi creates a memory. It drops entertainment audio (movies, TV shows, games, scripted fiction) and keeps real conversations, meetings, and educational content.

**Tier 1 — NLI (default on, local, fast)**
- Runs a zero-shot HuggingFace classifier inside the process
- Uses MPS on Apple Silicon, CUDA on Nvidia, CPU otherwise
- If confidence ≥ `NLI_THRESHOLD` (0.85) → decision is final
- If confidence < threshold → escalates to Tier 2

**Tier 2 — Ollama (default off)**
- HTTP POST to an Ollama instance with a one-shot prompt
- Only called when NLI is uncertain
- Times out after `OLLAMA_TIMEOUT` seconds → safe default: `keep`

**Safe defaults:** on any error (model failure, Ollama unreachable), the filter returns `keep` — content is never silently dropped.

---

## Benchmarking

```bash
# Quick benchmark — no diarization, no HF_TOKEN needed
python benchmark.py --no-diarization --no-embedding --language en --trials 1

# Full pipeline on real audio
python benchmark.py audio.wav --trials 3 --output json --output-file results_mac.json

# Compare two hardware profiles
python benchmark.py --compare results_mac.json results_rpi5.json
```

Or use the interactive UI at `http://localhost:8080/benchmark`.

---

## Hardware Support

| Hardware | Whisper | Diarization | Voice Encoder |
|---|---|---|---|
| Mac (Apple Silicon) | CPU + int8 | MPS | CPU |
| Linux + CUDA | CUDA + float16 | CUDA | CPU |
| Raspberry Pi 5 | CPU + int8 | CPU | CPU |
| Linux (CPU only) | CPU + int8 | CPU | CPU |

> CTranslate2 (Whisper backend) does not support MPS. resemblyzer does not support MPS. Both constraints are intentional — do not change them.

---

## Exposing to the Internet

The Omi pendant needs a public HTTPS URL. The recommended approach is a **Cloudflare Tunnel**:

```bash
cloudflared tunnel run --url http://localhost:8080
```

Or configure a named tunnel via `cloudflared-config.yml` for a stable hostname.

---

## Contributing

Issues and pull requests are welcome. A few guidelines:

- Do not add MPS support to the WhisperX call — CTranslate2 does not support it
- Do not add MPS support to `VoiceEncoder` — resemblyzer does not support it
- `load_align_model` must stay per-request (language varies per audio chunk)
- New pip dependencies must have `aarch64` wheels — the service must run on Raspberry Pi 5
- `classify_content()` must stay `async` — Ollama calls are async HTTP
- Do not call `_classify_with_nli()` directly from async code without `asyncio.to_thread`

---

## License

MIT

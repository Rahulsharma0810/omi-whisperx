# Agents Guide — omi-whisperx

Guidance for AI coding agents working on this repository.

## Project Overview

FastAPI service that transcribes audio using WhisperX, diarizes speakers, and resolves speaker names
via voice embeddings. Designed to run as a persistent background server connected to an Omi wearable
device via a Cloudflare tunnel.

## Key Files

| File | Purpose |
|---|---|
| `server.py` | Main FastAPI app — all transcription, diarization, speaker logic, and API routes |
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
                                                          └── /health     (model info)
```

Models are loaded **once at startup** (not per request):
- `whisperx.load_model` — transcription (CTranslate2, CPU+int8 or CUDA+float16)
- `DiarizationPipeline` — speaker diarization (MPS on Apple Silicon, CPU on RPi5)
- `VoiceEncoder` — speaker embeddings (always CPU — resemblyzer has no MPS support)

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
| `SPEAKER_THRESHOLD` | `0.80` | Cosine similarity threshold for speaker name matching |
| `WHISPER_INITIAL_PROMPT` | Hindi/English bilingual prompt | Primes Whisper for code-switching |
| `HF_TOKEN` | — | HuggingFace token for diarization model download |
| `PROFILES_DIR` | `~/.omi/speakers` | Speaker embedding storage directory |
| `CONTENT_FILTER` | `true` | Set to `false` to disable entertainment content filtering |

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
  1. Transcribe          model.transcribe(audio_path)         → segments + detected_lang
  2. Align               load_align_model() + whisperx.align() → word-level timestamps
  3. Diarize             diarize_model(audio_path)             → SPEAKER_00, SPEAKER_01 ...
  4. Assign speakers     assign_word_speakers()                → words labelled by speaker
  5. Embed               voice_encoder.embed_utterance()       → per-speaker numpy vectors
  6. Resolve names       cosine similarity vs stored profiles  → "Rahul" instead of SPEAKER_00
  7. Content filter      classify_content()                    → drop entertainment audio
  8. Return              JSON with segments, text, duration
```

## Content Filter

Added in commit `455ce7b`. Classifies transcripts as `entertainment` (gaming/TV) or `keep` before
Omi creates a memory. Returns empty segments for entertainment content.

- Built-in signals for common games (RDR2, Hitman) and generic phrases
- User-editable: `~/.omi/block_keywords.txt` and `~/.omi/allow_keywords.txt`
- `POST /filter/reload` — reload keyword files without restart
- Informative signals (≥2 hits) override entertainment detection — tutorials mentioning game names are kept

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
      WHISPER_MODEL: medium        # tiny/base/small/medium/large-v2
      WHISPER_BATCH_SIZE: "4"      # lower on RPi5 to save RAM
      SPEAKER_THRESHOLD: "0.80"
    volumes:
      - omi_data:/data             # model cache + speaker profiles
volumes:
  omi_data:
```

**Key files:**
- `Dockerfile` — multi-arch build, installs CPU-only torch on arm64
- `.github/workflows/docker.yml` — triggers on changes to server.py, requirements.txt, or Dockerfile
- `/data/huggingface` — HuggingFace model cache (mount a volume — models are ~2–4 GB)
- `/data/speakers` — speaker embedding profiles (persistent)

Models are **not** baked into the image. They download on first container start (~5 min).
Subsequent starts are fast once the volume is populated.

## Do Not

- Add MPS support to the whisperx transcription call — CTranslate2 does not support it
- Add MPS support to `VoiceEncoder` — resemblyzer does not support it
- Load `load_align_model` at startup — it must stay per-request (language varies per audio)
- Add new pip dependencies to `requirements.txt` without checking they have aarch64 wheels
  (the service must also run on Raspberry Pi 5)

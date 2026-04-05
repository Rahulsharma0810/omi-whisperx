import re
import json
import tempfile
import os
import asyncio
import logging
import time
from uuid import uuid4
import numpy as np
import torch
import httpx
from pathlib import Path
import platform
import resource
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from typing import Optional
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from resemblyzer import VoiceEncoder, preprocess_wav

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ntfy.sh alerts
# ---------------------------------------------------------------------------
NTFY_URL   = os.environ.get("NTFY_URL",   "https://ntfy.sh/ntfy-homeass-uptimekuma-topic")
NTFY_TOKEN = os.environ.get("NTFY_TOKEN", "")

# Simple rate-limiter: suppress identical alert titles within this window (seconds)
_NTFY_COOLDOWN = 120
_ntfy_last_sent: dict[str, float] = {}


async def notify(title: str, message: str, priority: str = "default", tags: str = "warning") -> None:
    """Fire-and-forget push to ntfy.sh. Suppresses duplicate titles within cooldown."""
    if not NTFY_TOKEN:
        return
    now = time.monotonic()
    last = _ntfy_last_sent.get(title, 0)
    if now - last < _NTFY_COOLDOWN:
        return
    _ntfy_last_sent[title] = now
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            await client.post(
                NTFY_URL,
                content=message.encode(),
                headers={
                    "Authorization": f"Bearer {NTFY_TOKEN}",
                    "Title": title,
                    "Priority": priority,
                    "Tags": tags,
                },
            )
    except Exception as e:
        logger.warning(f"[NTFY] Failed to send alert: {e}")

# ---------------------------------------------------------------------------
# SSE — live dashboard event bus
# ---------------------------------------------------------------------------
_sse_clients: set[asyncio.Queue] = set()

def emit_event(event_type: str, data: dict) -> None:
    """Push an event to every connected dashboard client."""
    payload = json.dumps({"type": event_type, **data})
    dead = set()
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    _sse_clients.difference_update(dead)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# App version: injected as APP_VERSION env var by CI (Docker build arg).
# Falls back to reading git SHA directly when running from source (dev/launchctl).
def _detect_version() -> str:
    v = os.environ.get("APP_VERSION", "").strip()
    if v and v != "dev":
        return v
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "dev"

APP_VERSION = _detect_version()

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "medium")
BATCH_SIZE = int(os.environ.get("WHISPER_BATCH_SIZE", "16"))
HF_TOKEN = os.environ.get("HF_TOKEN")
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.80"))
PROFILES_DIR = Path(os.environ.get("PROFILES_DIR", "~/.omi/speakers")).expanduser()

# Initial prompt for bilingual (Hindi+English / Hinglish) transcription.
# Override via WHISPER_INITIAL_PROMPT env var. The default primes Whisper
# to expect code-switching between Hindi and English.
INITIAL_PROMPT = os.environ.get(
    "WHISPER_INITIAL_PROMPT",
    "यह हिंदी और English में mixed conversation है। The speaker switches between Hindi and English freely.",
)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
# faster-whisper (CTranslate2) only supports "cpu" and "cuda" — NOT MPS.
# Use int8 on CPU (~3x faster than float32) or float16 on CUDA.
if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    WHISPER_DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

# PyTorch models (alignment, diarization) fully support MPS on Apple Silicon.
if torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
elif torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"

# Trigger phrase: "remember/save/call/name this voice as NAME"
CAPTURE_TRIGGER = re.compile(
    r"\b(?:remember|save|call|name)\s+this\s+voice\s+as\s+(.+)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Content filter — drops entertainment chunks so Omi doesn't save them
# Toggle: CONTENT_FILTER=false to disable entirely
# ---------------------------------------------------------------------------
CONTENT_FILTER_ENABLED = os.environ.get("CONTENT_FILTER", "true").lower() == "true"

# Tier 1 — Zero-shot NLI (fast, local, runs inside this process)
NLI_ENABLED = os.environ.get("NLI_ENABLED", "true").lower() == "true"
NLI_MODEL = os.environ.get("NLI_MODEL", "typeform/distilbert-base-uncased-mnli")
NLI_THRESHOLD = float(os.environ.get("NLI_THRESHOLD", "0.85"))

# Tier 2 — Ollama LLM (thorough, used only when NLI is uncertain)
# Enable: OLLAMA_ENABLED=true
# Point at your machine: OLLAMA_URL=http://192.168.1.X:11434
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "false").lower() == "true"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "5"))

# Descriptive labels — more words = better zero-shot accuracy
_NLI_LABELS = [
    "gaming, sitcom, movie, TV show, or scripted fiction",
    "tutorial, educational video, news, documentary, how-to, or real conversation",
]


def _classify_with_nli(text: str) -> tuple[str, float, list]:
    """
    Runs zero-shot NLI. Returns (decision, confidence, scores).
    decision is 'entertainment' or 'keep'.
    scores is a list of {label, score} for all candidate labels.
    """
    result = nli_pipeline(text[:400], candidate_labels=_NLI_LABELS)
    top_label: str = result["labels"][0]
    top_score: float = result["scores"][0]
    decision = "entertainment" if top_label == _NLI_LABELS[0] else "keep"
    scores = [
        {"label": l, "score": round(s, 3)}
        for l, s in zip(result["labels"], result["scores"])
    ]
    return decision, top_score, scores


_OLLAMA_RETRY_ATTEMPTS = 3
_OLLAMA_RETRY_DELAY   = 2.0  # seconds between retries


async def _classify_with_ollama(text: str) -> str:
    """
    Calls Ollama. Returns 'entertainment', 'keep', or 'keep' on any failure.
    Retries up to _OLLAMA_RETRY_ATTEMPTS times on transient ConnectError /
    TimeoutException before sending an ntfy alert.
    """
    prompt = (
        'Classify the audio transcript below as exactly one word:\n'
        '"entertainment" — gaming, sitcom, movie, TV show, or scripted fiction.\n'
        '"informative" — tutorial, educational video, news, documentary, how-to, or real conversation.\n\n'
        f'Transcript: """{text[:400]}"""\n\n'
        'Reply with one word only (entertainment or informative):'
    )

    last_exc: Exception | None = None
    last_exc_kind: str = ""

    for attempt in range(1, _OLLAMA_RETRY_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                )
                if resp.status_code == 429:
                    await notify("Ollama rate-limited (429)", f"Model: {OLLAMA_MODEL}\nURL: {OLLAMA_URL}", priority="high", tags="rotating_light")
                    logger.warning("[FILTER] Ollama 429 — keeping chunk")
                    return "keep"
                resp.raise_for_status()
                answer = resp.json().get("response", "").strip().lower()
                if "entertainment" in answer:
                    logger.info("[FILTER] Ollama → entertainment")
                    return "entertainment", answer
                if "informative" in answer:
                    logger.info("[FILTER] Ollama → informative (keep)")
                    return "keep", answer
                logger.warning(f"[FILTER] Ollama unclear: {answer!r} — keeping chunk")
                return "keep", answer
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_exc = e
            last_exc_kind = "unreachable" if isinstance(e, httpx.ConnectError) else "timeout"
            if attempt < _OLLAMA_RETRY_ATTEMPTS:
                logger.warning(
                    f"[FILTER] Ollama {last_exc_kind} (attempt {attempt}/{_OLLAMA_RETRY_ATTEMPTS}), "
                    f"retrying in {_OLLAMA_RETRY_DELAY}s…"
                )
                await asyncio.sleep(_OLLAMA_RETRY_DELAY)
            else:
                logger.warning(f"[FILTER] Ollama {last_exc_kind} after {_OLLAMA_RETRY_ATTEMPTS} attempts — keeping chunk")
        except Exception as e:
            await notify("Ollama error", f"{type(e).__name__}: {e}", priority="high", tags="x")
            logger.warning(f"[FILTER] Ollama error ({e}) — keeping chunk")
            return "keep", f"error: {e}"

    # All retries exhausted for ConnectError / TimeoutException
    if last_exc_kind == "unreachable":
        await notify(
            "Ollama unreachable",
            f"Cannot connect to {OLLAMA_URL} after {_OLLAMA_RETRY_ATTEMPTS} attempts\n{last_exc}",
            priority="high", tags="no_entry",
        )
        return "keep", "unreachable"
    else:
        await notify(
            "Ollama timeout",
            f"No response within {OLLAMA_TIMEOUT}s from {OLLAMA_URL} after {_OLLAMA_RETRY_ATTEMPTS} attempts",
            tags="hourglass_flowing_sand",
        )
        return "keep", "timeout"


async def classify_content(text: str, chunk_id: str) -> str:
    """
    Cascade classifier:
      Tier 1 — NLI: fast, local. If confident (≥ NLI_THRESHOLD) → done.
      Tier 2 — Ollama: only called when NLI is uncertain or disabled.
      Default → keep (never drop when unsure).
    Emits live events to the dashboard at each stage.
    """
    if not CONTENT_FILTER_ENABLED or not text.strip():
        return "keep"

    # Tier 1: NLI
    if NLI_ENABLED and nli_pipeline is not None:
        try:
            decision, confidence, scores = await asyncio.to_thread(_classify_with_nli, text)
        except Exception as e:
            await notify("NLI classifier failed", f"{type(e).__name__}: {e}", priority="high", tags="x")
            logger.error(f"[FILTER] NLI failed ({e}) — skipping to Ollama")
            decision, confidence, scores = "keep", 0.0, []
        confident = confidence >= NLI_THRESHOLD
        logger.info(f"[FILTER] NLI → {decision} ({confidence:.2f}) confident={confident}")
        emit_event("nli_result", {
            "chunk_id": chunk_id,
            "decision": decision,
            "confidence": round(confidence, 3),
            "confident": confident,
            "input_text": text[:200],
            "scores": scores,
        })
        if confident:
            return decision
        logger.info("[FILTER] NLI uncertain — escalating to Ollama")

    # Tier 2: Ollama
    if OLLAMA_ENABLED:
        result, raw = await _classify_with_ollama(text)
        emit_event("ollama_result", {
            "chunk_id": chunk_id,
            "decision": result,
            "input_text": text[:200],
            "raw_response": raw,
        })
        return result

    return "keep"

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
logger.info(
    f"Loading whisperx model '{MODEL_SIZE}' | "
    f"whisper={WHISPER_DEVICE}/{COMPUTE_TYPE} torch={TORCH_DEVICE}"
)
model = whisperx.load_model(MODEL_SIZE, WHISPER_DEVICE, compute_type=COMPUTE_TYPE)

logger.info("Loading diarization pipeline ...")
diarize_model = DiarizationPipeline(token=HF_TOKEN, device=TORCH_DEVICE)

logger.info("Loading speaker encoder ...")
# resemblyzer does not support MPS; keep on CPU
voice_encoder = VoiceEncoder(device="cpu")

# NLI zero-shot classifier (Tier 1 content filter)
nli_pipeline = None
if CONTENT_FILTER_ENABLED and NLI_ENABLED:
    from transformers import pipeline as hf_pipeline
    logger.info(f"Loading NLI classifier '{NLI_MODEL}' on {TORCH_DEVICE} ...")
    try:
        nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=NLI_MODEL,
            device=TORCH_DEVICE,
        )
        logger.info("NLI classifier ready.")
    except Exception as _nli_err:
        logger.error(f"NLI classifier failed to load: {_nli_err}")
        # Send synchronously at startup (no event loop yet)
        import threading
        def _startup_alert():
            import asyncio as _aio
            _aio.run(notify(
                "NLI failed to load",
                f"Model: {NLI_MODEL}\n{type(_nli_err).__name__}: {_nli_err}",
                priority="urgent", tags="x",
            ))
        threading.Thread(target=_startup_alert, daemon=True).start()


# ---------------------------------------------------------------------------
# Speaker registry
# ---------------------------------------------------------------------------
def load_profiles() -> dict[str, np.ndarray]:
    profiles = {}
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    for f in PROFILES_DIR.glob("*.npy"):
        if f.stem.startswith("__"):
            continue
        name = f.stem.replace("_", " ")
        profiles[name] = np.load(f)
        logger.info(f"Loaded speaker profile: {name}")
    return profiles


def save_profile(name: str, embedding: np.ndarray) -> None:
    filename = name.strip().replace(" ", "_") + ".npy"
    np.save(PROFILES_DIR / filename, embedding)
    logger.info(f"Saved speaker profile: {name}")


named_speakers: dict[str, np.ndarray] = load_profiles()
capture_pending: Optional[str] = None
recent_text_buffer: list[str] = []

logger.info("Models ready.")

app = FastAPI()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_hallucination(seg: dict) -> bool:
    text = seg.get("text", "").strip()
    if not text:
        return True
    words = text.split()
    if len(words) > 6 and len(set(words)) / len(words) < 0.3:
        return True
    return False


def get_speaker_embeddings(audio_path: str, diarize_segments) -> dict[str, np.ndarray]:
    """Extract per-speaker voice embeddings from diarized audio."""
    import soundfile as sf
    audio, sr = sf.read(audio_path)
    speaker_chunks: dict[str, list[np.ndarray]] = {}

    for _, row in diarize_segments.iterrows():
        label = row.get("speaker", "UNKNOWN")
        start = int(row["start"] * sr)
        end = int(row["end"] * sr)
        chunk = audio[start:end]
        if len(chunk) < sr:  # skip < 1s segments
            continue
        try:
            wav = preprocess_wav(chunk.astype(np.float32), source_sr=sr)
            emb = voice_encoder.embed_utterance(wav)
            speaker_chunks.setdefault(label, []).append(emb)
        except Exception:
            continue

    return {
        label: np.mean(embs, axis=0)
        for label, embs in speaker_chunks.items()
        if embs
    }


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def resolve_name(label: str, speaker_embeddings: dict[str, np.ndarray]) -> str:
    """Map SPEAKER_XX to a known name if similarity exceeds threshold."""
    emb = speaker_embeddings.get(label)
    if emb is None:
        return label

    best_name, best_score = None, 0.0
    for name, profile in named_speakers.items():
        score = similarity(profile, emb)
        if score > best_score:
            best_score = score
            best_name = name

    if best_name and best_score >= SPEAKER_THRESHOLD:
        return best_name

    return label


def check_capture_trigger(segments: list[dict]) -> Optional[str]:
    """Detect trigger across current + previous chunk to handle boundary splits."""
    current_text = " ".join(s.get("text", "") for s in segments)
    recent_text_buffer.append(current_text)
    if len(recent_text_buffer) > 2:
        recent_text_buffer.pop(0)

    match = CAPTURE_TRIGGER.search(" ".join(recent_text_buffer))
    if match:
        name = match.group(1).strip().title()
        recent_text_buffer.clear()
        return name
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
    response_format: Optional[str] = Form("verbose_json"),
):
    global capture_pending

    chunk_id = uuid4().hex[:8]
    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    emit_event("audio_received", {
        "chunk_id": chunk_id,
        "size_kb": round(len(audio_bytes) / 1024, 1),
    })

    try:
        # Transcribe — always, all speakers
        try:
            result = model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                batch_size=BATCH_SIZE,
            )
        except Exception as e:
            await notify("Transcription failed", f"chunk #{chunk_id}\n{type(e).__name__}: {e}", priority="urgent", tags="rotating_light")
            raise
        detected_lang = result.get("language", language or "hi")

        # Alignment: fall back to "hi" if detected language has no alignment model
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=TORCH_DEVICE
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio_path, TORCH_DEVICE
            )
        except Exception as e:
            logger.warning(f"Alignment failed for lang '{detected_lang}': {e} — skipping alignment")
            # segments still usable; just without word-level timestamps

        emit_event("transcribed", {
            "chunk_id": chunk_id,
            "lang": detected_lang,
            "segments": len(result.get("segments", [])),
        })

        diarize_segments = diarize_model(audio_path)
        result = assign_word_speakers(diarize_segments, result)
        segments = result.get("segments", [])

        # Extract per-speaker embeddings for name resolution
        speaker_embeddings = get_speaker_embeddings(audio_path, diarize_segments)

        # Check for "remember this voice as NAME" trigger
        triggered_name = check_capture_trigger(segments)
        if triggered_name:
            capture_pending = triggered_name
            logger.info(f"Capture mode: will enroll next new speaker as '{triggered_name}'")

        # Enroll the new speaker from this chunk if capture is pending
        if capture_pending and named_speakers.get(capture_pending) is None:
            for label, emb in speaker_embeddings.items():
                # Pick the first speaker that isn't already known
                already_known = any(
                    similarity(profile, emb) >= SPEAKER_THRESHOLD
                    for profile in named_speakers.values()
                )
                if not already_known:
                    named_speakers[capture_pending] = emb
                    save_profile(capture_pending, emb)
                    logger.info(f"Enrolled new speaker: '{capture_pending}'")
                    capture_pending = None
                    break

        # Build response
        full_text = " ".join(s.get("text", "").strip() for s in segments)
        duration = float(segments[-1]["end"]) if segments else 0.0

        # Content filter: drop entertainment chunks — Omi won't save empty segments
        content_class = await classify_content(full_text, chunk_id)
        if content_class == "entertainment":
            logger.info(f"[FILTER] Dropped entertainment content: {full_text[:120]!r}")
            emit_event("filtered", {
                "chunk_id": chunk_id,
                "preview": full_text[:80],
            })
            return JSONResponse({
                "task": "transcribe",
                "language": detected_lang,
                "duration": 0.0,
                "text": "",
                "segments": [],
            })

        formatted_segments = [
            {
                "id": i,
                "seek": 0,
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
                "text": seg.get("text", "").strip(),
                "speaker": resolve_name(seg.get("speaker", "UNKNOWN"), speaker_embeddings),
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": seg.get("avg_logprob", 0.0),
                "compression_ratio": seg.get("compression_ratio", 1.0),
                "no_speech_prob": seg.get("no_speech_prob", 0.0),
            }
            for i, seg in enumerate(segments)
            if not is_hallucination(seg)
        ]

        for seg in formatted_segments:
            logger.info(f"[{seg['speaker']}] {seg['start']}-{seg['end']}s: {seg['text']}")

        emit_event("transcript", {
            "chunk_id": chunk_id,
            "lang": detected_lang,
            "duration": duration,
            "segments": [
                {"speaker": s["speaker"], "text": s["text"],
                 "start": s["start"], "end": s["end"]}
                for s in formatted_segments
            ],
        })

        return JSONResponse({
            "task": "transcribe",
            "language": detected_lang,
            "duration": duration,
            "text": full_text,
            "segments": formatted_segments,
        })

    finally:
        os.unlink(audio_path)


# ---------------------------------------------------------------------------
# Benchmark — in-process pipeline timer reusing loaded models
# ---------------------------------------------------------------------------
_bench_clients: set[asyncio.Queue] = set()
_bench_running: bool = False
_bench_task: Optional[asyncio.Task] = None


def emit_bench(event_type: str, data: dict) -> None:
    payload = json.dumps({"type": event_type, **data})
    dead = set()
    for q in _bench_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    _bench_clients.difference_update(dead)


def _generate_bench_audio(duration: float, sample_rate: int = 16000) -> str:
    """Write speech-like synthetic audio to a temp WAV, return path."""
    import math
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    sig = sum(np.sin(2 * math.pi * f * t) for f in [120, 400, 1200, 2500])
    mod = 0.5 + 0.5 * np.sin(2 * math.pi * 4 * t)
    sig = sig * mod + rng.normal(0, 0.1, len(t))
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig / peak * 0.7
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, sig.astype(np.float32), sample_rate)
    return tmp.name


def _ram_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024 / 1024 if platform.system() == "Darwin" else usage.ru_maxrss / 1024


async def _run_bench(trials: int, language: str, no_alignment: bool,
                     no_diarization: bool, no_embedding: bool,
                     duration: float, provided_path: Optional[str] = None) -> None:
    global _bench_running
    audio_path = None
    try:
        if provided_path:
            audio_path = provided_path
            try:
                duration = sf.info(audio_path).duration
            except Exception:
                pass
        else:
            audio_path = await asyncio.to_thread(_generate_bench_audio, duration)

        audio_label = Path(provided_path).name if provided_path else f"{duration:.0f}s synthetic"
        emit_bench("bench_start", {
            "trials": trials, "audio_duration": duration, "language": language or "auto",
            "audio_label": audio_label,
            "system": {
                "whisper_device": WHISPER_DEVICE, "compute_type": COMPUTE_TYPE,
                "torch_device": TORCH_DEVICE, "model": MODEL_SIZE, "batch_size": BATCH_SIZE,
            },
        })

        all_times: dict[str, list[float]] = {}
        ram_baseline = _ram_mb()

        for trial in range(1, trials + 1):
            emit_bench("bench_trial_start", {"trial": trial, "total": trials})
            t_trial = time.perf_counter()
            timings: dict[str, float] = {}

            # --- Transcription ---
            t0 = time.perf_counter()
            result = await asyncio.to_thread(
                model.transcribe, audio_path,
                language=language or None, task="transcribe",
                batch_size=BATCH_SIZE,
            )
            timings["transcription"] = time.perf_counter() - t0
            detected_lang = result.get("language", language or "en")
            emit_bench("bench_stage", {"trial": trial, "stage": "transcription",
                                       "elapsed": round(timings["transcription"], 3)})

            # --- Alignment ---
            if not no_alignment:
                t0 = time.perf_counter()
                try:
                    align_model, metadata = await asyncio.to_thread(
                        whisperx.load_align_model, detected_lang, TORCH_DEVICE)
                    result = await asyncio.to_thread(
                        whisperx.align, result["segments"], align_model,
                        metadata, audio_path, TORCH_DEVICE)
                    del align_model
                except Exception as e:
                    logger.warning(f"[BENCH] Alignment skipped: {e}")
                timings["alignment"] = time.perf_counter() - t0
                emit_bench("bench_stage", {"trial": trial, "stage": "alignment",
                                           "elapsed": round(timings["alignment"], 3)})

            # --- Diarization ---
            diarize_segs = None
            if not no_diarization:
                t0 = time.perf_counter()
                try:
                    diarize_segs = await asyncio.to_thread(diarize_model, audio_path)
                    result = assign_word_speakers(diarize_segs, result)
                except Exception as e:
                    logger.warning(f"[BENCH] Diarization skipped: {e}")
                timings["diarization"] = time.perf_counter() - t0
                emit_bench("bench_stage", {"trial": trial, "stage": "diarization",
                                           "elapsed": round(timings["diarization"], 3)})

            # --- Embedding ---
            if not no_embedding and diarize_segs is not None:
                t0 = time.perf_counter()
                try:
                    audio_arr, sr = sf.read(audio_path)
                    for _, row in diarize_segs.iterrows():
                        chunk = audio_arr[int(row["start"] * sr):int(row["end"] * sr)]
                        if len(chunk) < sr:
                            continue
                        wav = preprocess_wav(chunk.astype(np.float32), source_sr=sr)
                        voice_encoder.embed_utterance(wav)
                except Exception as e:
                    logger.warning(f"[BENCH] Embedding skipped: {e}")
                timings["embedding"] = time.perf_counter() - t0
                emit_bench("bench_stage", {"trial": trial, "stage": "embedding",
                                           "elapsed": round(timings["embedding"], 3)})

            # --- Transcript + Content filter (NLI + Ollama) ---
            full_text = " ".join(
                seg.get("text", "").strip() for seg in result.get("segments", [])
            )
            emit_bench("bench_transcript", {
                "trial": trial,
                "text": full_text[:400],
                "lang": detected_lang,
                "segments": [
                    {"speaker": seg.get("speaker", "SPEAKER_00"),
                     "text": seg.get("text", "").strip(),
                     "start": round(seg.get("start", 0.0), 2),
                     "end": round(seg.get("end", 0.0), 2)}
                    for seg in result.get("segments", [])[:10]
                ],
            })

            if CONTENT_FILTER_ENABLED and full_text.strip():
                t0 = time.perf_counter()
                if NLI_ENABLED and nli_pipeline is not None:
                    try:
                        nli_dec, nli_conf, nli_scores = await asyncio.to_thread(
                            _classify_with_nli, full_text)
                        confident = nli_conf >= NLI_THRESHOLD
                        emit_bench("bench_nli_result", {
                            "trial": trial,
                            "decision": nli_dec,
                            "confidence": round(nli_conf, 3),
                            "confident": confident,
                            "input_text": full_text[:200],
                            "scores": nli_scores,
                        })
                        if not confident and OLLAMA_ENABLED:
                            oll_dec, oll_raw = await _classify_with_ollama(full_text)
                            emit_bench("bench_ollama_result", {
                                "trial": trial,
                                "decision": oll_dec,
                                "input_text": full_text[:200],
                                "raw_response": oll_raw,
                            })
                    except Exception as e:
                        logger.warning(f"[BENCH] Content filter failed: {e}")
                timings["content_filter"] = time.perf_counter() - t0
                emit_bench("bench_stage", {"trial": trial, "stage": "content_filter",
                                           "elapsed": round(timings["content_filter"], 3)})

            timings["total"] = time.perf_counter() - t_trial
            for k, v in timings.items():
                all_times.setdefault(k, []).append(v)
            emit_bench("bench_trial_done", {"trial": trial, "total": round(timings["total"], 3)})

        def _stats(name):
            times = all_times.get(name, [])
            if not times:
                return None
            mean = float(np.mean(times))
            std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
            return {"name": name, "times": [round(t, 3) for t in times],
                    "mean": round(mean, 3), "std": round(std, 3),
                    "rtf": round(mean / duration, 4)}

        stages = [s for s in [_stats(n) for n in
                               ["transcription", "alignment", "diarization",
                                "embedding", "content_filter"]] if s]
        total = _stats("total")
        emit_bench("bench_complete", {
            "stages": stages, "total": total,
            "ram_baseline_mb": round(ram_baseline, 1),
            "ram_peak_mb": round(_ram_mb(), 1),
        })

    except Exception as e:
        logger.error(f"[BENCH] {e}")
        emit_bench("bench_error", {"message": str(e)})
    finally:
        _bench_running = False
        if audio_path:
            try:
                os.unlink(audio_path)
            except OSError:
                pass


@app.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page():
    html_path = Path(__file__).parent / "benchmark.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/benchmark/events")
async def benchmark_sse(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _bench_clients.add(queue)

    async def stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            _bench_clients.discard(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/benchmark/run")
async def benchmark_run_endpoint(
    trials: int = Form(3),
    language: str = Form("en"),
    no_alignment: bool = Form(False),
    no_diarization: bool = Form(False),
    no_embedding: bool = Form(False),
    duration: float = Form(30.0),
    audio_file: Optional[UploadFile] = File(None),
):
    global _bench_running
    if _bench_running:
        return JSONResponse({"error": "Benchmark already running"}, status_code=409)
    _bench_running = True

    # Save uploaded audio to temp before the task starts (UploadFile closes after request)
    provided_path = None
    if audio_file and audio_file.filename:
        suffix = Path(audio_file.filename).suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(await audio_file.read())
        tmp.close()
        provided_path = tmp.name

    global _bench_task
    _bench_task = asyncio.create_task(_run_bench(
        trials, language, no_alignment, no_diarization, no_embedding,
        duration, provided_path))
    return JSONResponse({"status": "started"})


@app.get("/benchmark/status")
async def benchmark_status():
    return JSONResponse({"running": _bench_running})


@app.post("/benchmark/cancel")
async def benchmark_cancel():
    global _bench_running, _bench_task
    if _bench_task and not _bench_task.done():
        _bench_task.cancel()
    _bench_running = False
    emit_bench("bench_error", {"message": "cancelled"})
    return JSONResponse({"status": "cancelled"})


@app.get("/ui", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "ui.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/events")
async def sse():
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _sse_clients.add(queue)

    async def stream():
        try:
            while True:
                payload = await queue.get()
                yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _sse_clients.discard(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/speakers")
async def list_speakers():
    return {
        "named_speakers": list(named_speakers.keys()),
        "capture_pending": capture_pending,
    }


@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    key = name.replace("-", " ").replace("%20", " ")
    filename = PROFILES_DIR / (key.replace(" ", "_") + ".npy")
    named_speakers.pop(key, None)
    if filename.exists():
        filename.unlink()
    return JSONResponse({"status": "deleted", "name": key})


@app.delete("/speakers")
async def reset_all():
    global capture_pending
    capture_pending = None
    named_speakers.clear()
    recent_text_buffer.clear()
    for f in PROFILES_DIR.glob("*.npy"):
        if not f.stem.startswith("__"):
            f.unlink()
    return JSONResponse({"status": "reset"})


@app.get("/filter")
async def get_filter():
    return {
        "enabled": CONTENT_FILTER_ENABLED,
        "nli": {
            "enabled": NLI_ENABLED,
            "model": NLI_MODEL,
            "confidence_threshold": NLI_THRESHOLD,
            "loaded": nli_pipeline is not None,
        },
        "ollama": {
            "enabled": OLLAMA_ENABLED,
            "url": OLLAMA_URL,
            "model": OLLAMA_MODEL,
            "timeout_seconds": OLLAMA_TIMEOUT,
            "role": "fallback when NLI confidence < threshold",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "model": MODEL_SIZE,
        "named_speakers": list(named_speakers.keys()),
        "capture_pending": capture_pending,
        "speaker_threshold": SPEAKER_THRESHOLD,
        "content_filter": CONTENT_FILTER_ENABLED,
        "nli_enabled": NLI_ENABLED,
        "ollama_enabled": OLLAMA_ENABLED,
    }

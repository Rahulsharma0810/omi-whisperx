import re
import json
from collections import Counter
import struct
import tempfile
import os
import asyncio
import logging
import time
import socket
from urllib.parse import urlparse
from uuid import uuid4
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libtorchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*libavutil.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*resource_tracker.*", category=UserWarning)
import torch
import httpx
from pathlib import Path
import platform
import resource
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, Request, WebSocket, WebSocketDisconnect
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
# Suppress noisy third-party loggers
logging.getLogger("lightning.pytorch.utilities.upgrade_checkpoint").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)

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
_recent_events: list[str] = []  # last 50 events replayed to new SSE clients
_MAX_RECENT = 50

def emit_event(event_type: str, data: dict) -> None:
    """Push an event to every connected dashboard client and cache for replays."""
    payload = json.dumps({"type": event_type, **data})
    _recent_events.append(payload)
    if len(_recent_events) > _MAX_RECENT:
        _recent_events.pop(0)
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
SKIP_DIARIZE = os.environ.get("SKIP_DIARIZE", "false").lower() == "true"  # legacy: no speaker id at all
# Skip word-level alignment in live WebSocket path — saves ~2s, Omi only needs segment-level timestamps
SKIP_LIVE_ALIGN = os.environ.get("SKIP_LIVE_ALIGN", "true").lower() == "true"
# Trust client-side VAD (Omi VAD Gate) — skip server VAD, flush on any 0.5s frame gap
# Enable when iOS "VAD Gate" is ON — Omi already strips silence before sending
TRUST_CLIENT_VAD = os.environ.get("TRUST_CLIENT_VAD", "true").lower() == "true"
# Drop utterances that waited longer than this in the semaphore queue — prevents unbounded backlog
MAX_QUEUE_AGE = float(os.environ.get("MAX_QUEUE_AGE", "30"))
HF_TOKEN = os.environ.get("HF_TOKEN")
SPEAKER_THRESHOLD = float(os.environ.get("SPEAKER_THRESHOLD", "0.75"))
PROFILES_DIR = Path(os.environ.get("PROFILES_DIR", "~/.omi/speakers")).expanduser()
RECORDINGS_DIR = Path(os.environ.get("RECORDINGS_DIR", "~/.omi/recordings")).expanduser()
BLOCKED_DIR = Path(os.environ.get("BLOCKED_DIR", "~/.omi/blocked")).expanduser()
MAX_RECORDINGS = int(os.environ.get("MAX_RECORDINGS", "50"))
RECORDINGS_MAX_AGE_DAYS = int(os.environ.get("RECORDINGS_MAX_AGE_DAYS", "7"))
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
BLOCKED_DIR.mkdir(parents=True, exist_ok=True)

# Omi API — direct conversation creation, bypasses 2-min conversation_timeout
# POST segments immediately after speech ends instead of waiting for Omi backend timeout
OMI_API_KEY = os.environ.get("OMI_API_KEY", "")
OMI_API_BASE = os.environ.get("OMI_API_BASE", "https://api.omi.me/v1/dev/user")
OMI_USER_NAME = os.environ.get("OMI_USER_NAME", "")  # speaker name to mark as is_user=True
# Seconds of silence after last utterance before POSTing conversation to Omi API
OMI_CONV_DEBOUNCE = float(os.environ.get("OMI_CONV_DEBOUNCE", "30"))

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

# Trigger phrase — deliberate enrollment commands:
#   "remember/save/call/name this voice as NAME"
#   "recognize my voice as NAME"
CAPTURE_TRIGGER = re.compile(
    r"\b(?:"
    r"(?:remember|save|call|name)\s+this\s+voice\s+as"
    r"|recognize\s+(?:my\s+)?voice\s+as"
    r")\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Content filter — drops entertainment chunks so Omi doesn't save them
# Toggle: CONTENT_FILTER=false to disable entirely
# ---------------------------------------------------------------------------
CONTENT_FILTER_ENABLED = os.environ.get("CONTENT_FILTER", "false").lower() == "true"

# Tier 1 — Zero-shot NLI (fast, local, runs inside this process)
NLI_ENABLED = os.environ.get("NLI_ENABLED", "false").lower() == "true"
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


def _get_source_ip() -> str:
    """Return the local IP this host uses to reach external networks."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "unknown"


async def _tcp_probe(url: str, timeout: float = 3.0) -> bool:
    """Return True if the host:port in *url* accepts a TCP connection."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


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
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT, trust_env=False) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={"model": OLLAMA_MODEL,
                          "messages": [{"role": "user", "content": prompt}],
                          "stream": False},
                )
                if resp.status_code == 429:
                    await notify("Ollama rate-limited (429)", f"Model: {OLLAMA_MODEL}\nURL: {OLLAMA_URL}", priority="high", tags="rotating_light")
                    logger.warning("[FILTER] Ollama 429 — keeping chunk")
                    return "keep"
                resp.raise_for_status()
                answer = resp.json().get("message", {}).get("content", "").strip().lower()
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
            logger.warning(
                f"[FILTER] Ollama {last_exc_kind} (attempt {attempt}/{_OLLAMA_RETRY_ATTEMPTS}): {e}"
            )
            if attempt < _OLLAMA_RETRY_ATTEMPTS:
                await asyncio.sleep(_OLLAMA_RETRY_DELAY)
        except Exception as e:
            await notify("Ollama error", f"{type(e).__name__}: {e}", priority="high", tags="x")
            logger.warning(f"[FILTER] Ollama error ({e}) — keeping chunk")
            return "keep", f"error: {e}"

    # All retries exhausted for ConnectError / TimeoutException —
    # do a lightweight TCP probe before alerting to avoid false positives.
    source_ip = _get_source_ip()
    reachable = await _tcp_probe(OLLAMA_URL)
    exc_summary = f"{type(last_exc).__name__}: {last_exc}"
    if reachable:
        logger.warning(
            f"[FILTER] Ollama {last_exc_kind} after {_OLLAMA_RETRY_ATTEMPTS} attempts "
            f"but TCP probe succeeded — API may be starting up. URL: {OLLAMA_URL}"
        )
        return "keep", f"tcp_ok but api {last_exc_kind}: {exc_summary}"

    if last_exc_kind == "unreachable":
        await notify(
            "Ollama unreachable",
            f"Cannot connect to {OLLAMA_URL} after {_OLLAMA_RETRY_ATTEMPTS} attempts\n"
            f"Source IP: {source_ip}\n{last_exc}",
            priority="high", tags="no_entry",
        )
        return "keep", f"connect_refused: {OLLAMA_URL}"
    else:
        await notify(
            "Ollama timeout",
            f"No response within {OLLAMA_TIMEOUT}s from {OLLAMA_URL} after {_OLLAMA_RETRY_ATTEMPTS} attempts\n"
            f"Source IP: {source_ip}",
            tags="hourglass_flowing_sand",
        )
        return "keep", f"timeout ({OLLAMA_TIMEOUT}s): {OLLAMA_URL}"


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
from lightning_whisper_mlx import LightningWhisperMLX

logger.info(
    f"Loading lightning-whisper-mlx model '{MODEL_SIZE}' | "
    f"batch_size={BATCH_SIZE} torch={TORCH_DEVICE}"
)
model = LightningWhisperMLX(model=MODEL_SIZE, batch_size=BATCH_SIZE, quant=None)

logger.info("Loading diarization pipeline ...")
diarize_model = DiarizationPipeline(token=HF_TOKEN, device=TORCH_DEVICE)

# Cache align models per language — loading from disk per utterance was ~5-10s of the lag
_align_model_cache: dict = {}  # lang -> (align_model, metadata)

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
    """Save embedding. If a profile already exists, average with it for a more robust voiceprint."""
    filename = name.strip().replace(" ", "_") + ".npy"
    path = PROFILES_DIR / filename
    if path.exists():
        existing = np.load(str(path))
        embedding = (existing + embedding) / 2.0
        embedding /= np.linalg.norm(embedding)  # re-normalise after averaging
    np.save(str(path), embedding)
    named_speakers[name] = embedding  # keep in-memory copy in sync
    logger.info(f"Saved speaker profile: {name}")


named_speakers: dict[str, np.ndarray] = load_profiles()
capture_pending: Optional[str] = None
_omi_enroll: Optional[dict] = None  # {"name", "frames", "target", "done"}
recent_text_buffer: list[str] = []

logger.info("Models ready.")

app = FastAPI()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_hallucination(seg: dict) -> bool:
    import re
    text = seg.get("text", "").strip()
    if not text:
        return True

    # Segment shorter than 100 ms — too brief to be reliable
    if seg.get("end", 0.0) - seg.get("start", 0.0) < 0.1:
        return True

    # Whisper's own signal: high probability that no speech occurred
    if seg.get("no_speech_prob", 0.0) >= 0.45:
        return True

    # Low transcription confidence — Whisper was guessing
    if seg.get("avg_logprob", 0.0) < -0.8:
        return True

    # Only punctuation / symbols — no actual letters
    if not re.search(r"[a-zA-Z0-9\u0900-\u097F]", text):
        return True

    # Word-level repetition (e.g. "the the the the the the the")
    words = text.split()
    if len(words) > 6 and len(set(words)) / len(words) < 0.3:
        return True

    # Phrase-level repetition (e.g. "It's hot in the evening. It's hot in the evening.")
    # Check if any n-gram (n=4..6) repeats 3+ times
    for n in (4, 5, 6):
        if len(words) >= n * 3:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            most_common_count = Counter(ngrams).most_common(1)[0][1]
            if most_common_count >= 3:
                return True

    return False


# Short filler phrases from ambient TV/radio that slip past VAD
_TV_FILLER = re.compile(
    r"^\s*(thank you\.?|thanks\.?|thank you very much\.?|you'?re welcome\.?|"
    r"goodbye\.?|bye\.?|see you\.?|have a good day\.?|good night\.?|"
    r"welcome\.?|excuse me\.?|sorry\.?|please\.?|okay\.?|ok\.?|yes\.?|no\.?|"
    r"mm+\.?|hmm+\.?|uh+\.?|ah+\.?)\s*$",
    re.IGNORECASE,
)

def is_tv_filler(text: str, speaker: str) -> bool:
    """Return True for UNKNOWN short filler phrases likely from ambient TV/radio."""
    return speaker == "UNKNOWN" and bool(_TV_FILLER.match(text.strip()))


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


def _is_blocked(embedding: np.ndarray, threshold: float) -> bool:
    """Return True if this embedding matches any blocked voice."""
    for p in BLOCKED_DIR.glob("*.npy"):
        try:
            if similarity(np.load(str(p)), embedding) >= threshold:
                return True
        except Exception:
            pass
    return False


def _expire_old_recordings() -> None:
    """Delete unassigned clips older than RECORDINGS_MAX_AGE_DAYS."""
    cutoff = time.time() - RECORDINGS_MAX_AGE_DAYS * 86400
    for meta_path in list(RECORDINGS_DIR.glob("*.json")):
        if meta_path.stat().st_mtime < cutoff:
            rec_id = meta_path.stem
            for ext in (".json", ".wav", ".npy"):
                (RECORDINGS_DIR / f"{rec_id}{ext}").unlink(missing_ok=True)
            logger.info(f"[REC] Expired old clip {rec_id}")


def save_speaker_recording(audio_path: str, label: str, diarize_segments, chunk_id: str, embedding: np.ndarray) -> None:
    """Save one audio clip per unique unknown voice. Skip if blocked, enrolled, or already have 5 samples."""
    try:
        dedup_threshold = max(0.72, SPEAKER_THRESHOLD - 0.08)

        # Skip blocked voices (TV actors, recurring background voices, etc.)
        if _is_blocked(embedding, dedup_threshold):
            return

        # Skip if this voice is already enrolled as a named speaker
        for profile in named_speakers.values():
            if similarity(profile, embedding) >= dedup_threshold:
                return

        # Allow up to 5 recordings per unique unknown voice for better coverage
        same_voice_count = 0
        for emb_path in RECORDINGS_DIR.glob("*.npy"):
            try:
                existing = np.load(str(emb_path))
                if similarity(existing, embedding) >= dedup_threshold:
                    same_voice_count += 1
                    if same_voice_count >= 5:
                        return  # already have 5 samples of this voice
            except Exception:
                pass

        # Expire old clips before saving new ones
        _expire_old_recordings()

        raw = whisperx.load_audio(audio_path)  # float32 mono 16kHz
        sr = 16000
        # Collect up to 20s of this speaker's audio
        chunks = []
        total = 0.0
        for _, row in diarize_segments.iterrows():
            if row.get("speaker") != label:
                continue
            start = int(row["start"] * sr)
            end = int(row["end"] * sr)
            seg = raw[start:end]
            if len(seg) < sr * 0.5:  # skip < 0.5s
                continue
            chunks.append(seg)
            total += len(seg) / sr
            if total >= 20.0:
                break
        if not chunks:
            return
        if total < 5.0:  # skip clips too short for a reliable embedding
            logger.debug(f"[REC] Skipping {label}: only {total:.1f}s of audio (need ≥5s)")
            return
        audio_clip = np.concatenate(chunks)

        rec_id = uuid4().hex[:12]
        wav_path = RECORDINGS_DIR / f"{rec_id}.wav"
        meta_path = RECORDINGS_DIR / f"{rec_id}.json"
        emb_path = RECORDINGS_DIR / f"{rec_id}.npy"

        sf.write(str(wav_path), audio_clip, sr)
        np.save(str(emb_path), embedding)
        meta = {
            "id": rec_id,
            "speaker_label": label,
            "chunk_id": chunk_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "duration": round(total, 1),
        }
        meta_path.write_text(json.dumps(meta))

        # Evict oldest recordings if over limit
        all_meta = sorted(RECORDINGS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(all_meta) > MAX_RECORDINGS:
            old = all_meta.pop(0)
            rec_id_old = old.stem
            for ext in (".json", ".wav", ".npy"):
                (RECORDINGS_DIR / f"{rec_id_old}{ext}").unlink(missing_ok=True)

        logger.info(f"[REC] Saved {total:.1f}s clip for unknown speaker '{label}' → {rec_id}")
    except Exception as e:
        logger.warning(f"[REC] Failed to save recording for '{label}': {e}")


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def resolve_name(label: str, speaker_embeddings: dict[str, np.ndarray]) -> str:
    """Map SPEAKER_XX to a known name if similarity exceeds threshold."""
    emb = speaker_embeddings.get(label)
    if emb is None:
        logger.info(f"[SPEAKER] {label}: no embedding extracted")
        return label

    scores = {name: similarity(profile, emb) for name, profile in named_speakers.items()}
    if scores:
        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]
        logger.info(f"[SPEAKER] {label}: best={best_name} {best_score:.2f} "
                    + " ".join(f"{n}={s:.2f}" for n, s in scores.items()))
        if best_score >= SPEAKER_THRESHOLD:
            return best_name

    return label


def _fast_identify_speaker(audio_path: str) -> str:
    """Identify speaker by embedding whole utterance — no pyannote, ~0.1s.
    Saves clip for unknown voices. Returns resolved name or 'UNKNOWN'."""
    try:
        audio, sr = sf.read(audio_path)
    except Exception:
        return "UNKNOWN"

    if len(audio) < sr * 0.5:  # too short to embed reliably
        return "UNKNOWN"

    try:
        wav = preprocess_wav(audio.astype(np.float32), source_sr=sr)
        emb = voice_encoder.embed_utterance(wav)
    except Exception as e:
        logger.debug(f"[SPEAKER] embed failed: {e}")
        return "UNKNOWN"

    # Check blocked voices
    dedup_threshold = max(0.72, SPEAKER_THRESHOLD - 0.08)
    if _is_blocked(emb, dedup_threshold):
        logger.debug("[SPEAKER] utterance matches blocked voice — skipping")
        return "BLOCKED"

    # Match against enrolled speakers
    scores = {name: similarity(profile, emb) for name, profile in named_speakers.items()}
    if scores:
        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]
        logger.info(f"[SPEAKER] fast-id: best={best_name} {best_score:.2f} "
                    + " ".join(f"{n}={s:.2f}" for n, s in scores.items()))
        if best_score >= SPEAKER_THRESHOLD:
            return best_name

    # Unknown voice — save recording for later enrollment
    duration = len(audio) / sr
    if duration >= 5.0:
        try:
            _expire_old_recordings()
            # Dedup against existing unknown recordings
            same_voice_count = sum(
                1 for p in RECORDINGS_DIR.glob("*.npy")
                if similarity(np.load(str(p)), emb) >= dedup_threshold
            )
            if same_voice_count < 5:
                rec_id = uuid4().hex[:12]
                sf.write(str(RECORDINGS_DIR / f"{rec_id}.wav"), audio, sr)
                np.save(str(RECORDINGS_DIR / f"{rec_id}.npy"), emb)
                (RECORDINGS_DIR / f"{rec_id}.json").write_text(json.dumps({
                    "id": rec_id, "speaker_label": "UNKNOWN",
                    "chunk_id": rec_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "duration": round(duration, 1),
                }))
                logger.info(f"[REC] Saved {duration:.1f}s fast-id clip → {rec_id}")
        except Exception as e:
            logger.warning(f"[REC] fast-id save failed: {e}")

    return "UNKNOWN"


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
# WebSocket /live — Custom STT provider with diarization for Omi app
# ---------------------------------------------------------------------------

def _ogg_crc(data: bytes) -> int:
    """Ogg CRC-32 (polynomial 0x04c11db7, no reflection, no final XOR)."""
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04c11db7) if (crc & 0x80000000) else (crc << 1)
        crc &= 0xFFFFFFFF
    return crc


def _make_ogg_page(payload: bytes, granule: int, serial: int, seq: int, flags: int) -> bytes:
    """Build one Ogg page with correct CRC."""
    # Lacing: split payload into ≤255-byte segments
    segs = [payload[i:i + 255] for i in range(0, max(len(payload), 1), 255)]
    lacing = bytes(len(s) for s in segs)
    body = b"".join(segs)
    # Header with CRC zeroed
    hdr = (b"OggS" + struct.pack("<B", 0) + struct.pack("<B", flags)
           + struct.pack("<q", granule) + struct.pack("<I", serial)
           + struct.pack("<I", seq) + b"\x00\x00\x00\x00"
           + struct.pack("<B", len(segs)) + lacing)
    page = hdr + body
    crc = _ogg_crc(page)
    return page[:22] + struct.pack("<I", crc) + page[26:]


def _opus_frames_to_ogg(frames: list[bytes], sample_rate: int) -> bytes:
    """Wrap raw Opus frames in a valid Ogg-Opus container."""
    import struct as _struct
    serial = 0xA1B2C3D4

    # ID header
    id_hdr = (b"OpusHead" + _struct.pack("<B", 1) + _struct.pack("<B", 1)
              + _struct.pack("<H", 312) + _struct.pack("<I", sample_rate)
              + _struct.pack("<h", 0) + _struct.pack("<B", 0))
    # Comment header
    vendor = b"whisperx"
    com_hdr = (b"OpusTags" + _struct.pack("<I", len(vendor)) + vendor
               + _struct.pack("<I", 0))

    pages = [
        _make_ogg_page(id_hdr,  0, serial, 0, 0x02),  # BOS
        _make_ogg_page(com_hdr, 0, serial, 1, 0x00),
    ]
    # Opus always reports granule positions in 48 kHz samples; 20 ms = 960 samples
    granule = 0
    for i, frame in enumerate(frames):
        granule += 960  # 20 ms frame at 48 kHz
        eos = 0x04 if i == len(frames) - 1 else 0x00
        pages.append(_make_ogg_page(frame, granule, serial, i + 2, eos))

    return b"".join(pages)


def _decode_ws_audio(frames: list[bytes], codec: str, sample_rate: int) -> str:
    """Decode WebSocket audio frames to a WAV temp file. Returns the path."""
    import wave

    raw_bytes = b"".join(frames)

    # Already Ogg-containerised (magic OggS) — save as-is
    if raw_bytes[:4] == b"OggS":
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            f.write(raw_bytes)
            return f.name

    # Detect raw PCM16.
    # Omi always sends PCM16 at 16 kHz regardless of what the codec field says.
    # We use a majority-vote heuristic: sample 8 int16 values across the buffer;
    # if most are < 32000 (not clipped) it's almost certainly PCM, not Opus.
    step = max(2, len(raw_bytes) // 16) & ~1  # even step to stay on sample boundary
    samples = [
        abs(int.from_bytes(raw_bytes[i:i+2], "little", signed=True))
        for i in range(0, min(len(raw_bytes), step * 8), step)
    ]
    is_pcm = (codec.lower() in ("pcm", "pcm16", "pcm_s16le", "linear16")
              or sum(s < 32000 for s in samples) > len(samples) // 2)

    if is_pcm:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(raw_bytes)
        return path

    # Raw Opus frames — wrap in Ogg-Opus container
    ogg_data = _opus_frames_to_ogg(frames, sample_rate)
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        f.write(ogg_data)
        return f.name


async def _process_live_audio(
    frames: list[bytes], codec: str, sample_rate: int, language: Optional[str]
) -> list[dict]:
    """Decode frames → transcribe → (align) → (diarize) → return speaker-labelled segments."""
    t0 = time.time()
    audio_path = await asyncio.to_thread(_decode_ws_audio, frames, codec, sample_rate)

    try:
        result = await asyncio.to_thread(
            model.transcribe, audio_path,
            language=language,
        )
        t_transcribe = time.time()
        detected_lang = result.get("language", language or "hi")

        if SKIP_LIVE_ALIGN:
            t_align = time.time()  # skip — word timestamps not needed for Omi live path
        else:
            try:
                if detected_lang not in _align_model_cache:
                    logger.info(f"[WS] Loading align model '{detected_lang}' (first time, caching)...")
                    _align_model_cache[detected_lang] = await asyncio.to_thread(
                        whisperx.load_align_model, detected_lang, TORCH_DEVICE
                    )
                align_model, metadata = _align_model_cache[detected_lang]
                result = await asyncio.to_thread(
                    whisperx.align, result["segments"], align_model, metadata, audio_path, TORCH_DEVICE
                )
            except Exception as e:
                logger.warning(f"[WS] Alignment skipped: {e}")
            t_align = time.time()

        segments = result.get("segments", [])
        # Full pyannote diarization — per-segment speaker labels
        diarize_segs = await asyncio.to_thread(diarize_model, audio_path)
        result_with_speakers = assign_word_speakers(diarize_segs, result)
        segments = result_with_speakers.get("segments", [])
        speaker_embeddings = await asyncio.to_thread(get_speaker_embeddings, audio_path, diarize_segs)
        t_diarize = time.time()
        formatted = []
        for seg in segments:
            if is_hallucination(seg):
                continue
            label = seg.get("speaker", "UNKNOWN")
            resolved = resolve_name(label, speaker_embeddings)
            if resolved == "BLOCKED":
                continue
            seg_text = seg.get("text", "").strip()
            if is_tv_filler(seg_text, resolved):
                logger.debug(f"[WS] TV filler dropped: {seg_text!r}")
                continue
            formatted.append({
                "text": seg_text,
                "speaker": resolved,
                "start": round(seg.get("start", 0.0), 2),
                "end": round(seg.get("end", 0.0), 2),
            })

        logger.info(
            f"[WS] {len(formatted)} segs | "
            f"transcribe={t_transcribe-t0:.2f}s align={t_align-t_transcribe:.2f}s "
            f"diarize={t_diarize-t_align:.2f}s total={t_diarize-t0:.2f}s"
        )
        return formatted

    finally:
        os.unlink(audio_path)


# VAD-triggered utterance constants
_VAD_SILENCE_TRIGGER = 25   # 500 ms of silence (25 × 20 ms frames) ends an utterance
_VAD_MIN_SPEECH_FRAMES = 15  # 300 ms minimum utterance before we bother processing
_VAD_MAX_UTT_FRAMES = 1500   # 30s hard cap — covers virtually all natural sentences


_RMS_SILENCE_THRESHOLD = 50   # int16 RMS below this = true silence (was 300, too high for pendant)


def _compute_rms(frames: list[bytes]) -> float:
    """Return RMS energy of 640-byte PCM16 frames."""
    raw = b"".join(f for f in frames if len(f) == 640)
    if not raw:
        return 0.0
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples ** 2)))


def _has_speech_webrtcvad(frames: list[bytes], sample_rate: int) -> bool:
    """Return True only if frames have enough energy AND webrtcvad detects speech."""
    import webrtcvad

    # RMS energy gate — skip true silence
    rms = _compute_rms(frames)
    if rms < _RMS_SILENCE_THRESHOLD:
        logger.debug(f"[VAD] Skipping: rms={rms:.1f} < {_RMS_SILENCE_THRESHOLD}")
        return False

    vad = webrtcvad.Vad(1)  # aggressiveness 1 = least strict
    speech_frames = 0
    valid_frames = 0
    for frame in frames:
        if len(frame) == 640:  # 20ms at 16kHz
            valid_frames += 1
            try:
                if vad.is_speech(frame, sample_rate):
                    speech_frames += 1
            except Exception:
                pass
    if valid_frames == 0:
        return False
    # Require 40% of frames to be speech
    return speech_frames >= max(1, valid_frames * 4 // 10)


_omi_rate_limit_until: float = 0.0  # epoch seconds — don't POST before this time

async def _post_segments_to_omi(segments: list[dict], language: str, started_at: float, finished_at: float) -> None:
    """POST accumulated transcript segments to Omi API to create a conversation immediately.
    Bypasses Omi's 2-min conversation_timeout — conversation appears in app within seconds.
    """
    global _omi_rate_limit_until
    if not OMI_API_KEY or not segments:
        return
    if time.time() < _omi_rate_limit_until:
        remaining = int(_omi_rate_limit_until - time.time())
        logger.debug(f"[OMI] Rate-limited — skipping post, {remaining}s remaining")
        return
    import datetime, json as _json, urllib.request

    payload = {
        "transcript_segments": [
            {
                "text": s["text"],
                "speaker": s.get("speaker", "SPEAKER_00"),
                "is_user": s.get("speaker") == OMI_USER_NAME if OMI_USER_NAME else True,
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
            }
            for s in segments
            if s.get("text", "").strip()
        ],
        "source": "phone",
        "language": language or "en",
        "started_at": datetime.datetime.utcfromtimestamp(started_at).isoformat() + "Z",
        "finished_at": datetime.datetime.utcfromtimestamp(finished_at).isoformat() + "Z",
    }
    if not payload["transcript_segments"]:
        return

    def _do_post() -> None:
        global _omi_rate_limit_until
        data = _json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{OMI_API_BASE}/conversations/from-segments",
            data=data,
            headers={"Authorization": f"Bearer {OMI_API_KEY}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = _json.loads(resp.read())
                conv_id = body.get("id", "?")
                logger.info(f"[OMI] Conversation created: {conv_id} ({len(payload['transcript_segments'])} segs)")
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            logger.warning(f"[OMI] API error {e.code}: {body}")
            if e.code == 429:
                # Parse "Try again in Xs" from response body
                m = re.search(r"(\d+)\s*s", body)
                wait = int(m.group(1)) if m else 60
                _omi_rate_limit_until = time.time() + wait
                logger.warning(f"[OMI] Rate limited — backing off {wait}s")
        except Exception as e:
            logger.warning(f"[OMI] Failed to post conversation: {e}")

    await asyncio.to_thread(_do_post)


@app.websocket("/live")
async def live_transcription(
    websocket: WebSocket,
    uid: Optional[str] = None,
    language: Optional[str] = None,
    sample_rate: int = 16000,
    codec: str = "opus",
):
    await websocket.accept()
    logger.info(f"[WS] Connected uid={uid} lang={language} sr={sample_rate} codec={codec}")
    emit_event("ws_connected", {"uid": uid, "lang": language, "codec": codec})

    import webrtcvad as _webrtcvad
    vad = _webrtcvad.Vad(1)  # aggressiveness 1 = least strict (3 was filtering real speech)

    all_segments: list = []
    utterance_tasks: list = []
    ws_lock = asyncio.Lock()
    proc_sem = asyncio.Semaphore(1)  # one utterance through WhisperX at a time

    # Omi API direct posting — accumulate segments, POST after OMI_CONV_DEBOUNCE silence
    omi_pending_segs: list = []
    omi_conv_started_at: float = 0.0
    omi_debounce_task: Optional[asyncio.Task] = None

    async def _omi_debounce_fire() -> None:
        await asyncio.sleep(OMI_CONV_DEBOUNCE)
        nonlocal omi_pending_segs, omi_conv_started_at
        if omi_pending_segs:
            segs = omi_pending_segs[:]
            t_start = omi_conv_started_at
            omi_pending_segs = []
            omi_conv_started_at = 0.0
            await _post_segments_to_omi(segs, language, t_start, time.time())

    def _omi_reset_debounce(new_segs: list) -> None:
        nonlocal omi_pending_segs, omi_conv_started_at, omi_debounce_task
        if new_segs:
            if not omi_pending_segs:
                omi_conv_started_at = time.time()
            omi_pending_segs.extend(new_segs)
        if omi_debounce_task and not omi_debounce_task.done():
            omi_debounce_task.cancel()
        if omi_pending_segs:
            omi_debounce_task = asyncio.create_task(_omi_debounce_fire())

    # VAD state
    speech_buffer: list[bytes] = []
    silence_count = 0
    speaking = False
    utterance_start_frame = 0
    utterance_wall_start: float = 0.0  # wall-clock when first speech frame of utterance received
    frame_count = 0   # total frames received (for timestamp calculation)
    session_start = time.time()

    async def _flush_utterance(frames: list[bytes], offset: float, audio_recv_at: float) -> None:
        """Process one VAD-detected utterance: transcribe, send to Omi, emit SSE."""
        queued_at = time.time()
        if not TRUST_CLIENT_VAD:
            has_speech = await asyncio.to_thread(_has_speech_webrtcvad, frames, sample_rate)
            if not has_speech:
                logger.debug(f"[WS] Utterance @{offset:.1f}s skipped (silence)")
                return
        async with proc_sem:  # one at a time — prevents MPS OOM from parallel tasks
            proc_start = time.time()
            queue_wait = proc_start - queued_at
            if queue_wait > MAX_QUEUE_AGE:
                logger.warning(
                    f"[WS] Utterance @{offset:.1f}s DROPPED — waited {queue_wait:.1f}s "
                    f"(MAX_QUEUE_AGE={MAX_QUEUE_AGE}s)"
                )
                return
            try:
                segs = await _process_live_audio(frames, codec, sample_rate, language)
                sent_at = time.time()
                proc_time = sent_at - proc_start
                total_lag = sent_at - audio_recv_at
                for seg in segs:
                    seg["start"] = round(seg["start"] + offset, 2)
                    seg["end"] = round(seg["end"] + offset, 2)
                if not segs:
                    return
                all_segments.extend(segs)
                _omi_reset_debounce(segs)  # restart 30s timer; POST to Omi API when silence detected
                emit_event("ws_transcript", {"uid": uid, "segments": segs,
                    "lag": {"audio_recv_at": round(audio_recv_at - session_start, 2),
                            "sent_at": round(sent_at - session_start, 2),
                            "queue_wait_s": round(queue_wait, 2),
                            "proc_time_s": round(proc_time, 2),
                            "total_lag_s": round(total_lag, 2)}})
                # Omi iOS requires: no type field, OR type="Results". Any other type = ignored.
                payload = json.dumps({
                    "segments": segs,
                })
                async with ws_lock:
                    try:
                        await websocket.send_text(payload)
                    except Exception:
                        pass
                logger.info(
                    f"[WS] Utterance @{offset:.1f}s → {len(segs)} seg(s) | "
                    f"audio_recv={time.strftime('%H:%M:%S', time.localtime(audio_recv_at))}.{int(audio_recv_at % 1 * 1000):03d} "
                    f"sent={time.strftime('%H:%M:%S', time.localtime(sent_at))}.{int(sent_at % 1 * 1000):03d} "
                    f"queue_wait={queue_wait:.2f}s proc={proc_time:.2f}s total_lag={total_lag:.2f}s | "
                    + " | ".join(f"[{s['speaker']}] {s['text']}" for s in segs)
                )
            except Exception as e:
                err = str(e)
                if "No active speech" not in err and "Unspecified internal error" not in err:
                    logger.error(f"[WS] Utterance @{offset:.1f}s failed: {e}")
            finally:
                # Release MPS memory after each utterance
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

    def _schedule_utterance() -> None:
        """Snapshot current speech_buffer, reset VAD state, fire async task."""
        nonlocal speech_buffer, silence_count, speaking, utterance_start_frame, utterance_wall_start
        frames = speech_buffer[:]
        offset = utterance_start_frame * 0.02
        recv_at = utterance_wall_start  # wall-clock of first speech frame
        speech_buffer = []
        silence_count = 0
        speaking = False
        utterance_wall_start = 0.0
        task = asyncio.create_task(_flush_utterance(frames, offset, recv_at))
        utterance_tasks.append(task)

    # Timeout to flush utterance when Omi VAD is on (silence frames not sent)
    _FRAME_GAP_TIMEOUT = 0.5  # seconds between frames before we treat gap as utterance end

    try:
        while True:
            # TRUST_CLIENT_VAD: Omi VAD Gate already stripped silence — always use short gap timeout
            if TRUST_CLIENT_VAD:
                recv_timeout = _FRAME_GAP_TIMEOUT if speech_buffer else 90.0
            else:
                recv_timeout = _FRAME_GAP_TIMEOUT if speaking else 90.0
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=recv_timeout)
            except asyncio.TimeoutError:
                has_buffer = len(speech_buffer) >= _VAD_MIN_SPEECH_FRAMES
                if (speaking or TRUST_CLIENT_VAD) and has_buffer:
                    logger.debug("[WS] Frame gap — flushing utterance")
                    _schedule_utterance()
                elif not speaking and not (TRUST_CLIENT_VAD and speech_buffer):
                    logger.warning("[WS] Timed out after 90s inactivity")
                    break
                continue

            if message["type"] == "websocket.disconnect":
                break

            if message.get("bytes"):
                frame = message["bytes"]
                if frame_count == 0:
                    logger.info(f"[WS] First audio frame: {len(frame)} bytes | hex: {frame[:8].hex()}")
                    emit_event("ws_audio", {"uid": uid, "frame_bytes": len(frame)})

                # Tap into Omi enrollment capture if active
                global _omi_enroll
                if _omi_enroll and not _omi_enroll.get("done"):
                    _omi_enroll["frames"].append(frame)
                    collected = len(_omi_enroll["frames"])
                    target = _omi_enroll["target"]
                    if collected % 50 == 0:  # progress every 1s
                        emit_event("omi_enroll_progress", {
                            "name": _omi_enroll["name"],
                            "pct": round(collected / target * 100),
                            "secs": collected // 50,
                            "total": target // 50,
                        })
                    if collected >= target:
                        _omi_enroll["done"] = True
                        asyncio.create_task(_finish_omi_enrollment(_omi_enroll.copy()))
                        _omi_enroll = None

                if TRUST_CLIENT_VAD:
                    # Client already stripped silence — every frame is speech
                    if not speaking:
                        speaking = True
                        utterance_start_frame = frame_count
                        utterance_wall_start = time.time()
                    speech_buffer.append(frame)
                    # Hard cap: flush at 30s regardless
                    if len(speech_buffer) >= _VAD_MAX_UTT_FRAMES:
                        _schedule_utterance()
                else:
                    is_speech = False
                    if len(frame) == 640:
                        try:
                            is_speech = vad.is_speech(frame, sample_rate)
                        except Exception:
                            pass

                    # Debug log every 50 frames (1s)
                    if frame_count % 50 == 0:
                        rms = 0
                        if len(frame) == 640:
                            samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
                            rms = int(np.sqrt(np.mean(samples ** 2)))
                        logger.debug(
                            f"[WS] frame={frame_count} size={len(frame)} rms={rms} "
                            f"is_speech={is_speech} speaking={speaking} buf={len(speech_buffer)}"
                        )

                    if is_speech:
                        if not speaking:
                            speaking = True
                            utterance_start_frame = frame_count
                            utterance_wall_start = time.time()
                        silence_count = 0
                        speech_buffer.append(frame)
                    else:
                        if speaking:
                            silence_count += 1
                            speech_buffer.append(frame)
                            if (silence_count >= _VAD_SILENCE_TRIGGER
                                    and len(speech_buffer) >= _VAD_MIN_SPEECH_FRAMES):
                                _schedule_utterance()
                            elif len(speech_buffer) >= _VAD_MAX_UTT_FRAMES:
                                _schedule_utterance()
                        else:
                            speech_buffer.append(frame)
                            if len(speech_buffer) >= 250:  # 5s force-flush
                                rms = _compute_rms(speech_buffer)
                                if rms < _RMS_SILENCE_THRESHOLD:
                                    logger.debug(f"[WS] 5s buffer silent (rms={rms:.1f}) — discarding")
                                speech_buffer = []
                            else:
                                logger.info(f"[WS] VAD never triggered — force-flushing {len(speech_buffer)} frames (rms={rms:.1f})")
                                utterance_start_frame = frame_count - len(speech_buffer)
                                utterance_wall_start = time.time() - len(speech_buffer) * 0.02
                                _schedule_utterance()

                frame_count += 1

            elif message.get("text"):
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "CloseStream":
                        logger.info(f"[WS] CloseStream — {frame_count} frames total")
                        break
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"[WS] Error: {e}")

    emit_event("ws_disconnected", {"uid": uid, "frames": frame_count})

    # Flush any remaining speech
    if speaking and len(speech_buffer) >= _VAD_MIN_SPEECH_FRAMES:
        _schedule_utterance()

    if utterance_tasks:
        await asyncio.gather(*utterance_tasks, return_exceptions=True)

    # Cancel debounce timer and POST any remaining segments immediately on disconnect
    if omi_debounce_task and not omi_debounce_task.done():
        omi_debounce_task.cancel()
    if omi_pending_segs:
        await _post_segments_to_omi(omi_pending_segs, language, omi_conv_started_at, time.time())

    try:
        await websocket.close()
    except Exception:
        pass


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
            )
        except Exception as e:
            await notify("Transcription failed", f"chunk #{chunk_id}\n{type(e).__name__}: {e}", priority="urgent", tags="rotating_light")
            raise
        detected_lang = result.get("language", language or "hi")

        # Alignment: fall back to "hi" if detected language has no alignment model
        try:
            if detected_lang not in _align_model_cache:
                logger.info(f"Loading align model for '{detected_lang}' (first time, caching)...")
                _align_model_cache[detected_lang] = whisperx.load_align_model(
                    language_code=detected_lang, device=TORCH_DEVICE
                )
            align_model, metadata = _align_model_cache[detected_lang]
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

        # Save clips for unrecognized speakers so user can label them later
        for label, emb in speaker_embeddings.items():
            if resolve_name(label, speaker_embeddings) == label:  # still anonymous
                save_speaker_recording(audio_path, label, diarize_segments, chunk_id, emb)

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
                language=language or None,
            )
            timings["transcription"] = time.perf_counter() - t0
            detected_lang = result.get("language", language or "en")
            emit_bench("bench_stage", {"trial": trial, "stage": "transcription",
                                       "elapsed": round(timings["transcription"], 3)})

            # --- Alignment ---
            if not no_alignment:
                t0 = time.perf_counter()
                try:
                    if detected_lang not in _align_model_cache:
                        _align_model_cache[detected_lang] = await asyncio.to_thread(
                            whisperx.load_align_model, detected_lang, TORCH_DEVICE)
                    align_model, metadata = _align_model_cache[detected_lang]
                    result = await asyncio.to_thread(
                        whisperx.align, result["segments"], align_model,
                        metadata, audio_path, TORCH_DEVICE)
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
                    # whisperx.load_audio handles any ffmpeg-supported format (m4a, mp3, etc.)
                    raw = whisperx.load_audio(audio_path)  # float32 mono at 16kHz
                    sr = 16000
                    audio_arr = raw
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
    # Replay recent events so new clients catch up
    for payload in list(_recent_events):
        queue.put_nowait(payload)
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
    names = list(named_speakers.keys())
    quality = {}
    for name, emb in named_speakers.items():
        others = {n: e for n, e in named_speakers.items() if n != name}
        if others:
            scores = {n: float(similarity(e, emb)) for n, e in others.items()}
            closest_name = max(scores, key=scores.__getitem__)
            closest_score = scores[closest_name]
            gap = round(1.0 - closest_score, 3)
            # Rating: excellent ≥0.30, good 0.18-0.30, fair 0.10-0.18, poor <0.10
            rating = "excellent" if gap >= 0.30 else "good" if gap >= 0.18 else "fair" if gap >= 0.10 else "poor"
        else:
            gap, closest_name, closest_score, rating = None, None, None, "only"
        quality[name] = {"gap": gap, "closest": closest_name, "closest_score": round(closest_score, 3) if closest_score else None, "rating": rating}
    return {
        "named_speakers": names,
        "quality": quality,
        "capture_pending": capture_pending,
    }


@app.get("/ui/live", response_class=HTMLResponse)
async def live_ui():
    p = Path(__file__).parent / "live.html"
    return HTMLResponse(p.read_text())


@app.get("/ui/speakers", response_class=HTMLResponse)
async def speakers_ui():
    p = Path(__file__).parent / "speakers.html"
    return HTMLResponse(p.read_text())


def _purge_matched_recordings() -> int:
    """Delete any saved recording whose embedding now matches an enrolled speaker."""
    dedup_threshold = max(0.72, SPEAKER_THRESHOLD - 0.08)
    removed = 0
    for emb_path in list(RECORDINGS_DIR.glob("*.npy")):
        try:
            emb = np.load(str(emb_path))
            for profile in named_speakers.values():
                if similarity(profile, emb) >= dedup_threshold:
                    rec_id = emb_path.stem
                    for ext in (".json", ".wav", ".npy"):
                        (RECORDINGS_DIR / f"{rec_id}{ext}").unlink(missing_ok=True)
                    removed += 1
                    break
        except Exception:
            pass
    return removed


@app.get("/speakers/recordings")
async def list_recordings():
    recs = []
    for meta_path in RECORDINGS_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text())
            emb_path = RECORDINGS_DIR / f"{meta['id']}.npy"
            if emb_path.exists() and named_speakers:
                emb = np.load(str(emb_path))
                scores = {n: round(similarity(p, emb), 3) for n, p in named_speakers.items()}
                best_name = max(scores, key=scores.__getitem__)
                best_score = scores[best_name]
                if best_score >= SPEAKER_THRESHOLD:
                    meta["best_match"] = {"name": best_name, "score": best_score, "all": scores}
                else:
                    meta["best_match"] = None
            else:
                meta["best_match"] = None
            recs.append(meta)
        except Exception:
            pass
    # Sort: highest similarity first, unmatched last
    recs.sort(key=lambda r: r["best_match"]["score"] if r["best_match"] else 0, reverse=True)
    return {"recordings": recs}


@app.get("/speakers/recordings/{rec_id}/audio")
async def get_recording_audio(rec_id: str):
    from fastapi.responses import FileResponse
    wav = RECORDINGS_DIR / f"{rec_id}.wav"
    if not wav.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(wav), media_type="audio/wav")


@app.post("/speakers/recordings/{rec_id}/assign")
async def assign_recording(rec_id: str, name: str = Form(...)):
    """Assign a name to a saved recording — enrolls the speaker and deletes the clip."""
    name = name.strip().title()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    emb_path = RECORDINGS_DIR / f"{rec_id}.npy"
    if not emb_path.exists():
        return JSONResponse({"error": "recording not found"}, status_code=404)
    emb = np.load(str(emb_path))
    named_speakers[name] = emb
    save_profile(name, emb)
    for ext in (".json", ".wav", ".npy"):
        (RECORDINGS_DIR / f"{rec_id}{ext}").unlink(missing_ok=True)
    purged = _purge_matched_recordings()
    logger.info(f"[REC] Assigned recording {rec_id} → '{name}', purged {purged} duplicate(s)")
    return JSONResponse({"status": "enrolled", "name": name, "purged": purged})


@app.post("/speakers/recordings/purge")
async def purge_recordings():
    """Delete all recordings that now match an enrolled speaker."""
    removed = _purge_matched_recordings()
    return JSONResponse({"purged": removed})


@app.post("/speakers/recordings/{rec_id}/block")
async def block_recording(rec_id: str):
    """Block this voice — save embedding to blocked list, delete clip. Never records this voice again."""
    emb_path = RECORDINGS_DIR / f"{rec_id}.npy"
    if not emb_path.exists():
        return JSONResponse({"error": "recording not found"}, status_code=404)
    emb = np.load(str(emb_path))

    # Save to blocked dir
    block_id = uuid4().hex[:12]
    np.save(str(BLOCKED_DIR / f"{block_id}.npy"), emb)

    # Delete the recording
    for ext in (".json", ".wav", ".npy"):
        (RECORDINGS_DIR / f"{rec_id}{ext}").unlink(missing_ok=True)

    # Also purge any other recordings of the same voice
    dedup_threshold = max(0.72, SPEAKER_THRESHOLD - 0.08)
    purged = 0
    for other_emb_path in list(RECORDINGS_DIR.glob("*.npy")):
        try:
            if similarity(np.load(str(other_emb_path)), emb) >= dedup_threshold:
                rid = other_emb_path.stem
                for ext in (".json", ".wav", ".npy"):
                    (RECORDINGS_DIR / f"{rid}{ext}").unlink(missing_ok=True)
                purged += 1
        except Exception:
            pass

    logger.info(f"[REC] Blocked voice {rec_id}, purged {purged} similar clip(s)")
    return JSONResponse({"status": "blocked", "purged": purged})


@app.get("/speakers/blocked")
async def list_blocked():
    count = len(list(BLOCKED_DIR.glob("*.npy")))
    return {"blocked_voices": count}


@app.delete("/speakers/blocked")
async def clear_blocked():
    removed = 0
    for p in list(BLOCKED_DIR.glob("*.npy")):
        p.unlink(missing_ok=True)
        removed += 1
    return JSONResponse({"cleared": removed})


@app.delete("/speakers/recordings/{rec_id}")
async def delete_recording(rec_id: str):
    for ext in (".json", ".wav", ".npy"):
        (RECORDINGS_DIR / f"{rec_id}{ext}").unlink(missing_ok=True)
    return JSONResponse({"status": "deleted"})


async def _finish_omi_enrollment(capture: dict) -> None:
    """Extract resemblyzer embedding from Omi-captured frames and save profile."""
    name = capture["name"]
    frames = capture["frames"]
    audio_path = await asyncio.to_thread(_decode_ws_audio, frames, "pcm", 16000)
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        wav = preprocess_wav(audio.astype(np.float32), source_sr=sr)
        emb = voice_encoder.embed_utterance(wav)
        named_speakers[name] = emb
        save_profile(name, emb)
        purged = _purge_matched_recordings()
        emit_event("omi_enroll_complete", {"name": name, "purged": purged})
        logger.info(f"[ENROLL-OMI] Enrolled '{name}' from {len(frames)} Omi frames, purged {purged}")
    except Exception as e:
        emit_event("omi_enroll_error", {"name": name, "error": str(e)})
        logger.error(f"[ENROLL-OMI] Failed for '{name}': {e}")
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass


@app.post("/speakers/enroll-omi")
async def start_omi_enrollment(name: str = Form(...), duration: int = Form(30)):
    """Begin capturing Omi audio for speaker enrollment."""
    global _omi_enroll
    name = name.strip().title()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    if _omi_enroll and not _omi_enroll.get("done"):
        return JSONResponse({"error": "Enrollment already in progress"}, status_code=409)
    target = duration * 50  # 20 ms frames → 50/s
    _omi_enroll = {"name": name, "frames": [], "target": target, "done": False}
    emit_event("omi_enroll_start", {"name": name, "duration": duration})
    logger.info(f"[ENROLL-OMI] Started capture for '{name}' ({duration}s)")
    return {"status": "capturing", "name": name, "duration": duration}


@app.delete("/speakers/enroll-omi")
async def cancel_omi_enrollment():
    global _omi_enroll
    _omi_enroll = None
    emit_event("omi_enroll_cancelled", {})
    return {"status": "cancelled"}


@app.post("/speakers")
async def enroll_speaker(name: str = Form(...), file: UploadFile = File(...)):
    """Enroll a speaker from any audio clip (WAV, M4A, MP3, etc.)."""
    name = name.strip().title()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)

    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        raw = whisperx.load_audio(audio_path)  # float32 mono 16kHz
        wav = preprocess_wav(raw.astype(np.float32), source_sr=16000)
        emb = voice_encoder.embed_utterance(wav)
    except Exception as e:
        os.unlink(audio_path)
        return JSONResponse({"error": f"embedding failed: {e}"}, status_code=500)
    finally:
        try:
            os.unlink(audio_path)
        except Exception:
            pass

    named_speakers[name] = emb
    save_profile(name, emb)
    purged = _purge_matched_recordings()
    logger.info(f"[SPEAKERS] Enrolled via upload: '{name}', purged {purged} matching recording(s)")
    return JSONResponse({"status": "enrolled", "name": name})


@app.patch("/speakers/{name}")
async def rename_speaker(name: str, new_name: str = Form(...)):
    """Rename an existing speaker profile."""
    old_key = name.replace("-", " ").replace("%20", " ").replace("_", " ")
    new_key = new_name.strip().title()
    if not new_key:
        return JSONResponse({"error": "new_name required"}, status_code=400)
    if old_key not in named_speakers:
        return JSONResponse({"error": "speaker not found"}, status_code=404)

    emb = named_speakers.pop(old_key)
    named_speakers[new_key] = emb

    old_file = PROFILES_DIR / (old_key.replace(" ", "_") + ".npy")
    if old_file.exists():
        old_file.unlink()
    save_profile(new_key, emb)
    logger.info(f"[SPEAKERS] Renamed '{old_key}' → '{new_key}'")
    return JSONResponse({"status": "renamed", "old": old_key, "new": new_key})


@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    key = name.replace("-", " ").replace("%20", " ").replace("_", " ")
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
        "whisper": {
            "model": MODEL_SIZE,
            "batch_size": BATCH_SIZE,
            "device": WHISPER_DEVICE,
            "compute_type": COMPUTE_TYPE,
            "torch_device": TORCH_DEVICE,
        },
        "speakers": {
            "named": list(named_speakers.keys()),
            "capture_pending": capture_pending,
            "threshold": SPEAKER_THRESHOLD,
            "profiles_dir": str(PROFILES_DIR),
        },
        "content_filter": {
            "enabled": CONTENT_FILTER_ENABLED,
            "nli": {
                "enabled": NLI_ENABLED,
                "model": NLI_MODEL,
                "threshold": NLI_THRESHOLD,
                "loaded": nli_pipeline is not None,
            },
            "ollama": {
                "enabled": OLLAMA_ENABLED,
                "url": OLLAMA_URL,
                "model": OLLAMA_MODEL,
                "timeout": OLLAMA_TIMEOUT,
            },
        },
        "notifications": {
            "ntfy_url": NTFY_URL,
        },
    }

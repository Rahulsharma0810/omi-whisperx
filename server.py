import re
import tempfile
import os
import logging
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
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
# Config
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------
CONTENT_FILTER_ENABLED = os.environ.get("CONTENT_FILTER", "true").lower() == "true"
FILTER_DIR = Path(os.environ.get("FILTER_DIR", "~/.omi")).expanduser()

# Built-in entertainment signals (need ≥2 hits to trigger, avoids false positives)
_ENTERTAINMENT_SIGNALS: list[str] = [
    # Red Dead Redemption 2
    "arthur morgan", "dutch van der linde", "hosea matthews", "micah bell",
    "javier escuella", "john marston", "red dead redemption",
    "blackwater", "saint denis", "strawberry", "valentine",
    # Hitman / Agent 47
    "agent 47", "diana burnwood", "elusive target", "silent assassin",
    "contracts mode",
    # Generic game-mechanics phrases
    "mission failed", "mission complete", "mission passed",
    "game over", "respawn", "checkpoint reached",
    "achievement unlocked", "press x to", "press a to",
    "health regenerating", "wanted level",
    "new objective", "follow the marker",
]

# Informative / educational signals — ≥2 hits keep the chunk regardless
_INFORMATIVE_SIGNALS: list[str] = [
    "in this video", "in today's video", "in this tutorial",
    "let me show you", "let me explain", "let me walk you through",
    "how to", "step by step", "the reason why", "which means that",
    "in other words", "for example", "for instance",
    "according to", "research shows", "studies show",
    "the key takeaway", "important point",
    "next step", "let's dive into", "let's talk about",
    "tutorial", "explained", "guide", "tips and tricks",
    "if you found this helpful", "hope this helps",
]


def _load_keyword_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def _load_filter_lists() -> tuple[list[str], list[str]]:
    block = _load_keyword_file(FILTER_DIR / "block_keywords.txt")
    allow = _load_keyword_file(FILTER_DIR / "allow_keywords.txt")
    if block:
        logger.info(f"Loaded {len(block)} custom block keywords")
    if allow:
        logger.info(f"Loaded {len(allow)} custom allow keywords")
    return block, allow


_user_block, _user_allow = _load_filter_lists()


def classify_content(text: str) -> str:
    """
    Returns 'entertainment' if the transcript is gaming/scripted fiction,
    otherwise 'keep'. Default is always 'keep' when ambiguous.
    """
    if not CONTENT_FILTER_ENABLED or not text.strip():
        return "keep"

    t = text.lower()

    # User allow-list overrides everything (e.g. "RDR2 speedrun tips")
    if any(kw in t for kw in _user_allow):
        return "keep"

    # ≥2 informative signals → educational content, keep regardless
    if sum(1 for s in _INFORMATIVE_SIGNALS if s in t) >= 2:
        return "keep"

    # Hard block from user-defined file
    if any(kw in t for kw in _user_block):
        return "entertainment"

    # ≥2 built-in entertainment signals → filter
    if sum(1 for s in _ENTERTAINMENT_SIGNALS if s in t) >= 2:
        return "entertainment"

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

    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        # Transcribe — always, all speakers
        # initial_prompt primes Whisper for Hindi/English code-switching
        result = model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            batch_size=BATCH_SIZE,
            initial_prompt=INITIAL_PROMPT if not language else None,
        )
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
        content_class = classify_content(full_text)
        if content_class == "entertainment":
            logger.info(f"[FILTER] Dropped entertainment content: {full_text[:120]!r}")
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

        return JSONResponse({
            "task": "transcribe",
            "language": detected_lang,
            "duration": duration,
            "text": full_text,
            "segments": formatted_segments,
        })

    finally:
        os.unlink(audio_path)


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
        "block_keywords": _user_block,
        "allow_keywords": _user_allow,
        "block_file": str(FILTER_DIR / "block_keywords.txt"),
        "allow_file": str(FILTER_DIR / "allow_keywords.txt"),
    }


@app.post("/filter/reload")
async def reload_filter():
    """Reload block/allow keyword files without restarting the server."""
    global _user_block, _user_allow
    _user_block, _user_allow = _load_filter_lists()
    return {
        "status": "reloaded",
        "block_count": len(_user_block),
        "allow_count": len(_user_allow),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_SIZE,
        "named_speakers": list(named_speakers.keys()),
        "capture_pending": capture_pending,
        "speaker_threshold": SPEAKER_THRESHOLD,
        "content_filter": CONTENT_FILTER_ENABLED,
    }

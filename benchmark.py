#!/usr/bin/env python3
"""
benchmark.py — WhisperX pipeline benchmarking for omi-whisperx.

Replicates the exact server.py inference pipeline and times each stage.
Outputs Real-Time Factor (RTF) per stage for comparing hardware (e.g. Mac M2 vs RPi5).

Usage examples:
  # Quick run (no diarization, synthetic audio):
  python benchmark.py --no-diarization --no-embedding --language en --trials 1

  # Full pipeline with real audio:
  python benchmark.py audio.wav --trials 3 --output json --output-file results_mac.json

  # Compare two machines:
  python benchmark.py --compare results_mac.json results_rpi5.json

NOTE: When using synthetic audio (no AUDIO_FILE given), use --language en to prevent
language-detection errors. Diarization requires HF_TOKEN env var or --hf-token.
"""

import argparse
import json
import logging
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import whisperx
from resemblyzer import VoiceEncoder, preprocess_wav
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device detection — exact mirror of server.py lines 41-56
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    WHISPER_DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    WHISPER_DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

if torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
elif torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name: str
    times_sec: list
    mean_sec: float
    std_sec: float
    rtf: float
    rtf_std: float


@dataclass
class BenchmarkResult:
    timestamp: str
    system_info: dict
    audio: dict
    config: dict
    stages: list
    total: dict
    memory: dict


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def get_process_ram_mb() -> float:
    """Current RSS in MB, cross-platform."""
    if platform.system() == "Linux":
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except OSError:
            pass
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / 1024 / 1024  # bytes -> MB
    return usage.ru_maxrss / 1024  # kB -> MB


def maybe_empty_accel_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


class MpsPeakMemoryTracker:
    """Background-thread MPS memory peak tracker (MPS has no native peak API)."""

    def __init__(self, poll_interval: float = 0.05):
        self._interval = poll_interval
        self._peak = 0
        self._running = False
        self._thread = None

    def __enter__(self):
        if not torch.backends.mps.is_available():
            return self
        self._peak = torch.mps.driver_allocated_memory()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while self._running:
            cur = torch.mps.driver_allocated_memory()
            if cur > self._peak:
                self._peak = cur
            time.sleep(self._interval)

    def __exit__(self, *_):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> Optional[float]:
        if not torch.backends.mps.is_available():
            return None
        return self._peak / 1024 / 1024


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------
def get_cpu_model() -> str:
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        else:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
                    if line.lower().startswith("hardware"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_total_ram_gb() -> Optional[float]:
    try:
        if platform.system() == "Darwin":
            r = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3,
            )
            if r.returncode == 0:
                return round(int(r.stdout.strip()) / 1024 ** 3, 1)
        else:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / 1024 ** 2, 1)
    except Exception:
        pass
    return None


def collect_system_info(model_size: str, batch_size: int) -> dict:
    import importlib.metadata
    try:
        wx_ver = importlib.metadata.version("whisperx")
    except Exception:
        wx_ver = "unknown"
    return {
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu_model": get_cpu_model(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": get_total_ram_gb(),
        "whisper_device": WHISPER_DEVICE,
        "compute_type": COMPUTE_TYPE,
        "torch_device": TORCH_DEVICE,
        "torch_version": torch.__version__,
        "whisperx_version": wx_ver,
        "python_version": platform.python_version(),
        "model_size": model_size,
        "batch_size": batch_size,
    }


# ---------------------------------------------------------------------------
# Synthetic audio generation
# ---------------------------------------------------------------------------
def generate_synthetic_wav(duration_secs: float, output_path: str, sample_rate: int = SAMPLE_RATE):
    """
    Generate speech-like noise that passes pyannote VAD.
    Uses multi-band AM noise with syllable-rate modulation (~4 Hz).
    """
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration_secs, int(sample_rate * duration_secs), endpoint=False)

    # Formant-like bands: F0, F1, F2, F3
    freqs = [120, 400, 1200, 2500]
    signal = np.zeros_like(t)
    for f in freqs:
        signal += np.sin(2 * math.pi * f * t)

    # Syllable-rate amplitude modulation
    mod = 0.5 + 0.5 * np.sin(2 * math.pi * 4 * t)
    signal *= mod

    # Add background noise at -20 dB
    signal += rng.normal(0, 0.1, len(t))

    # Normalise to -3 dBFS
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.7

    sf.write(output_path, signal.astype(np.float32), sample_rate)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_models(model_size, hf_token, skip_diarization, skip_embedding, verbose):
    if verbose:
        logging.getLogger().setLevel(logging.INFO)

    print(f"Loading whisperx model '{model_size}' [{WHISPER_DEVICE}/{COMPUTE_TYPE}] ...", flush=True)
    model = whisperx.load_model(model_size, WHISPER_DEVICE, compute_type=COMPUTE_TYPE)

    diarize_model = None
    if not skip_diarization:
        print(f"Loading diarization pipeline [{TORCH_DEVICE}] ...", flush=True)
        try:
            diarize_model = DiarizationPipeline(token=hf_token, device=TORCH_DEVICE)
        except Exception as e:
            print(
                f"ERROR: Failed to load diarization model: {e}\n"
                f"Tip: set HF_TOKEN env var or pass --hf-token, or use --no-diarization",
                file=sys.stderr,
            )
            sys.exit(1)

    voice_encoder = None
    if not skip_embedding:
        print("Loading voice encoder [cpu] ...", flush=True)
        voice_encoder = VoiceEncoder(device="cpu")  # resemblyzer: CPU only

    return model, diarize_model, voice_encoder


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------
def run_single_trial(audio_path, model, diarize_model, voice_encoder, args) -> dict:
    timings = {}

    # --- Stage 1: Transcription ---
    t0 = time.perf_counter()
    result = model.transcribe(
        audio_path,
        language=args.language,
        task="transcribe",
        batch_size=args.batch_size,
        initial_prompt=None,  # no prompt in bench mode for reproducibility
    )
    timings["transcription"] = time.perf_counter() - t0
    detected_lang = result.get("language", args.language or "en")

    # --- Stage 2: Alignment ---
    if not args.no_alignment:
        t0 = time.perf_counter()
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=detected_lang, device=TORCH_DEVICE
            )
            result = whisperx.align(
                result["segments"], align_model, metadata, audio_path, TORCH_DEVICE
            )
            del align_model
            maybe_empty_accel_cache()
        except Exception as e:
            logger.warning(f"Alignment failed for '{detected_lang}': {e} — skipping")
        timings["alignment"] = time.perf_counter() - t0

    diarize_segments = None

    # --- Stage 3: Diarization ---
    if not args.no_diarization and diarize_model is not None:
        t0 = time.perf_counter()
        try:
            diarize_segments = diarize_model(audio_path)
            result = assign_word_speakers(diarize_segments, result)
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
        timings["diarization"] = time.perf_counter() - t0

    # --- Stage 4: Speaker embeddings ---
    if not args.no_embedding and voice_encoder is not None and diarize_segments is not None:
        t0 = time.perf_counter()
        try:
            audio_arr, sr = sf.read(audio_path)
            for _, row in diarize_segments.iterrows():
                start = int(row["start"] * sr)
                end = int(row["end"] * sr)
                chunk = audio_arr[start:end]
                if len(chunk) < sr:
                    continue
                wav = preprocess_wav(chunk.astype(np.float32), source_sr=sr)
                voice_encoder.embed_utterance(wav)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
        timings["embedding"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    return timings


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
def run_benchmark(audio_path, model, diarize_model, voice_encoder, args) -> BenchmarkResult:
    import datetime

    audio_info = sf.info(audio_path)
    audio_duration = audio_info.duration

    print(f"\nAudio: {Path(audio_path).name} | {audio_duration:.1f}s | {audio_info.samplerate} Hz")
    print(f"Model: {args.model} | batch={args.batch_size} | trials={args.trials} | warmup={'yes' if args.warmup else 'no'}")

    ram_baseline = get_process_ram_mb()

    # Warmup
    if args.warmup:
        print("Running warmup pass ...", flush=True)
        run_single_trial(audio_path, model, diarize_model, voice_encoder, args)
        maybe_empty_accel_cache()

    # Timed trials
    all_timings: dict[str, list] = {}
    ram_peak = ram_baseline
    mps_tracker = MpsPeakMemoryTracker()

    for i in range(args.trials):
        print(f"Trial {i + 1}/{args.trials} ...", end=" ", flush=True)
        t_start = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with mps_tracker:
            trial = run_single_trial(audio_path, model, diarize_model, voice_encoder, args)

        elapsed = time.perf_counter() - t_start
        ram_now = get_process_ram_mb()
        if ram_now > ram_peak:
            ram_peak = ram_now

        print(f"total={elapsed:.1f}s  RTF={trial['total'] / audio_duration:.3f}")
        for k, v in trial.items():
            all_timings.setdefault(k, []).append(v)

        maybe_empty_accel_cache()

    # Aggregate stats
    def make_stage(name) -> dict:
        times = all_timings.get(name, [])
        if not times:
            return None
        mean = float(np.mean(times))
        std = float(np.std(times, ddof=1)) if len(times) > 1 else 0.0
        return {
            "name": name,
            "times_sec": [round(t, 4) for t in times],
            "mean_sec": round(mean, 4),
            "std_sec": round(std, 4),
            "rtf": round(mean / audio_duration, 4),
            "rtf_std": round(std / audio_duration, 4),
        }

    stage_names = ["transcription", "alignment", "diarization", "embedding"]
    stages = [make_stage(s) for s in stage_names if make_stage(s) is not None]
    total = make_stage("total")

    # Memory
    accel_peak = None
    if torch.cuda.is_available():
        accel_peak = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    elif mps_tracker.peak_mb is not None:
        accel_peak = round(mps_tracker.peak_mb, 1)

    memory = {
        "ram_baseline_mb": round(ram_baseline, 1),
        "ram_peak_mb": round(ram_peak, 1),
        "ram_delta_mb": round(ram_peak - ram_baseline, 1),
        "accel_device": TORCH_DEVICE if TORCH_DEVICE != "cpu" else None,
        "accel_peak_mb": accel_peak,
    }

    return BenchmarkResult(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        system_info=collect_system_info(args.model, args.batch_size),
        audio={
            "file": Path(audio_path).name,
            "generated": getattr(args, "_generated_audio", False),
            "duration_sec": round(audio_duration, 3),
            "sample_rate": audio_info.samplerate,
        },
        config={
            "num_trials": args.trials,
            "warmup": args.warmup,
            "no_alignment": args.no_alignment,
            "no_diarization": args.no_diarization,
            "no_embedding": args.no_embedding,
            "language": args.language,
        },
        stages=stages,
        total=total,
        memory=memory,
    )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------
def format_text_report(result: BenchmarkResult) -> str:
    si = result.system_info
    mem = result.memory
    lines = []
    sep = "=" * 62

    lines.append(sep)
    lines.append(f"  omi-whisperx Benchmark — {result.timestamp}")
    lines.append(sep)
    lines.append("")
    lines.append("System")
    lines.append(f"  Platform   : {si['platform']} ({si['cpu_model']})")
    lines.append(f"  CPU cores  : {si['cpu_count']}  RAM: {si.get('total_ram_gb', '?')} GB")
    lines.append(f"  Devices    : whisper={si['whisper_device']}/{si['compute_type']}  torch={si['torch_device']}")
    lines.append(f"  PyTorch    : {si['torch_version']}  WhisperX: {si['whisperx_version']}")
    lines.append("")
    a = result.audio
    lines.append(f"Audio: {a['file']}{'  (generated)' if a['generated'] else ''}  {a['duration_sec']}s  {a['sample_rate']} Hz")
    c = result.config
    lines.append(f"Model: {si['model_size']} | batch={si['batch_size']} | trials={c['num_trials']} | warmup={'yes' if c['warmup'] else 'no'}")
    lines.append("")
    lines.append(f"{'Stage':<18} {'Mean(s)':>8} {'Std(s)':>8} {'RTF':>8} {'RTF Std':>8}")
    lines.append("-" * 58)
    for s in result.stages:
        lines.append(f"  {s['name']:<16} {s['mean_sec']:>8.3f} {s['std_sec']:>8.3f} {s['rtf']:>8.4f} {s['rtf_std']:>8.4f}")
    if result.total:
        t = result.total
        lines.append("-" * 58)
        lines.append(f"  {'TOTAL':<16} {t['mean_sec']:>8.3f} {t['std_sec']:>8.3f} {t['rtf']:>8.4f} {t['rtf_std']:>8.4f}")
    lines.append("")
    lines.append(f"RAM baseline: {mem['ram_baseline_mb']:.0f} MB | peak: {mem['ram_peak_mb']:.0f} MB | delta: {mem['ram_delta_mb']:.0f} MB")
    if mem.get("accel_peak_mb") is not None:
        lines.append(f"{mem['accel_device'].upper()} peak: {mem['accel_peak_mb']:.0f} MB")
    lines.append(sep)

    # Raw trial times
    if result.stages and result.config["num_trials"] > 1:
        lines.append("")
        lines.append("Raw trial times (seconds)")
        lines.append("-" * 58)
        n = result.config["num_trials"]
        header = f"{'Stage':<18}" + "".join(f"  Trial {i+1}" for i in range(n))
        lines.append(header)
        for s in result.stages:
            row = f"  {s['name']:<16}" + "".join(f"  {v:>7.3f}" for v in s["times_sec"])
            lines.append(row)
        if result.total:
            row = f"  {'total':<16}" + "".join(f"  {v:>7.3f}" for v in result.total["times_sec"])
            lines.append(row)

    return "\n".join(lines)


def format_json_report(result: BenchmarkResult) -> str:
    return json.dumps(asdict(result), indent=2)


def print_comparison(a: dict, b: dict):
    """Print side-by-side comparison of two JSON benchmark results."""
    a_si = a["system_info"]
    b_si = b["system_info"]
    a_label = f"{a_si['cpu_model'][:20]}"
    b_label = f"{b_si['cpu_model'][:20]}"

    print(f"\n{'Stage':<18} {a_label:>22} {b_label:>22} {'Speedup':>10}")
    print("-" * 76)

    a_stages = {s["name"]: s for s in a["stages"]}
    b_stages = {s["name"]: s for s in b["stages"]}

    all_names = list(dict.fromkeys(list(a_stages) + list(b_stages) + ["total"]))
    for name in all_names:
        sa = (a_stages.get(name) or a.get("total")) if name == "total" else a_stages.get(name)
        sb = (b_stages.get(name) or b.get("total")) if name == "total" else b_stages.get(name)
        if not sa or not sb:
            continue
        a_rtf = sa["rtf"]
        b_rtf = sb["rtf"]
        speedup = b_rtf / a_rtf if a_rtf > 0 else float("inf")
        label = "TOTAL" if name == "total" else name
        print(f"  {label:<16} {a_rtf:>20.4f} {b_rtf:>22.4f} {speedup:>9.1f}x")

    print()
    print(f"Audio duration: {a['audio']['duration_sec']}s")
    print(f"Model: {a_si['model_size']} | batch={a_si['batch_size']}")
    print(f"RTF < 1.0 = faster than real-time")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("audio", nargs="?", metavar="AUDIO_FILE",
                   help="Audio file to benchmark. Omit to generate synthetic audio.")
    p.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                   help="Compare two JSON result files side-by-side.")
    p.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "medium"),
                   help="Whisper model size (default: medium)")
    p.add_argument("--batch-size", type=int,
                   default=int(os.environ.get("WHISPER_BATCH_SIZE", "16")),
                   help="WhisperX batch size (default: 16)")
    p.add_argument("--trials", type=int, default=3,
                   help="Number of timed trials (default: 3)")
    p.add_argument("--language", default=None,
                   help="Force language code, e.g. en, hi. Recommended with synthetic audio.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for diarization model. Falls back to $HF_TOKEN.")
    p.add_argument("--no-diarization", action="store_true",
                   help="Skip diarization stage (use when no HF_TOKEN)")
    p.add_argument("--no-alignment", action="store_true",
                   help="Skip alignment stage")
    p.add_argument("--no-embedding", action="store_true",
                   help="Skip speaker embedding stage")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Synthetic audio duration in seconds (default: 30)")
    p.add_argument("--no-warmup", dest="warmup", action="store_false", default=True,
                   help="Disable warmup pass")
    p.add_argument("--output", choices=["text", "json"], default="text",
                   help="Output format (default: text)")
    p.add_argument("--output-file", default=None,
                   help="Write results to file (default: stdout)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose model loading logs")
    return p.parse_args()


def main():
    args = parse_args()

    # Compare mode — no model loading needed
    if args.compare:
        with open(args.compare[0]) as f:
            a = json.load(f)
        with open(args.compare[1]) as f:
            b = json.load(f)
        print_comparison(a, b)
        return

    # Resolve audio file
    tmp_audio = None
    if args.audio:
        audio_path = args.audio
        args._generated_audio = False
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        tmp_audio = tmp.name
        print(f"Generating {args.duration}s synthetic audio -> {tmp_audio}", flush=True)
        generate_synthetic_wav(args.duration, tmp_audio)
        audio_path = tmp_audio
        args._generated_audio = True
        if not args.language:
            print("WARN: No --language set with synthetic audio. Defaulting to 'en'.", flush=True)
            args.language = "en"

    try:
        model, diarize_model, voice_encoder = load_models(
            args.model, args.hf_token,
            args.no_diarization, args.no_embedding,
            args.verbose,
        )

        result = run_benchmark(audio_path, model, diarize_model, voice_encoder, args)

        if args.output == "json":
            output = format_json_report(result)
        else:
            output = format_text_report(result)

        if args.output_file:
            Path(args.output_file).write_text(output)
            print(f"\nResults written to {args.output_file}")
        else:
            print("\n" + output)

    finally:
        if tmp_audio:
            try:
                os.unlink(tmp_audio)
            except OSError:
                pass


if __name__ == "__main__":
    main()

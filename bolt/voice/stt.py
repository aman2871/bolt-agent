"""Speech to text via whisper.cpp, run as a subprocess."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

# Same models directory as the TTS voices, so one variable moves both.
MODELS_DIR = Path(os.environ.get(
    "BOLT_MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))
MODEL = Path(os.environ.get("BOLT_WHISPER_MODEL", MODELS_DIR / "ggml-base.en.bin"))
BINARY = os.environ.get("BOLT_WHISPER_BIN", "whisper-cli")
SAMPLE_RATE = 16_000
# Whisper labels non-speech audio in brackets — "[BLANK_AUDIO]", "(keyboard
# clicking)", "[ Writing on Board ]". None of it is ever something said to us, and
# an always-on mic produces a lot of it.
_BRACKETED = re.compile(r"[\[(][^\]\)]{0,60}[\])]")


class STTUnavailable(RuntimeError):
    pass


def available() -> tuple[bool, str]:
    """Check presence only. Running the binary compiles Metal shaders on first
    use, which takes far longer than a health check should."""
    if not MODEL.is_file():
        return False, f"whisper model missing at {MODEL}"
    if shutil.which(BINARY) is None:
        return False, f"{BINARY} not found (brew install whisper-cpp)"
    return True, "ready"


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Write mono 16-bit PCM, converting from float32 if needed."""
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(samples.tobytes())
    return path


def transcribe_file(wav_path: Path | str, threads: int = 4) -> str:
    ok, reason = available()
    if not ok:
        raise STTUnavailable(reason)

    proc = subprocess.run(
        [BINARY, "-m", str(MODEL), "-f", str(wav_path),
         "--no-timestamps", "--no-prints", "--language", "en", "--threads", str(threads)],
        capture_output=True, text=True, timeout=180,
    )
    if proc.returncode != 0:
        raise STTUnavailable(f"whisper failed: {proc.stderr[-400:]}")

    text = _BRACKETED.sub(" ", proc.stdout)
    cleaned = " ".join(text.split()).strip()
    # If the annotation was the whole utterance, there was no speech in it.
    return "" if cleaned in ("", ".", "-", "...") else cleaned


def transcribe(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe an in-memory mono buffer."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    try:
        write_wav(path, samples, sample_rate)
        return transcribe_file(path)
    finally:
        path.unlink(missing_ok=True)

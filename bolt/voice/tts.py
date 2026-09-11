"""Text to speech via Kokoro, streamed a sentence at a time."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Iterator

import numpy as np

MODELS = Path(os.environ.get(
    "BOLT_MODELS_DIR", Path(__file__).resolve().parents[2] / "models"))
MODEL_PATH = MODELS / "kokoro-v1.0.onnx"
VOICES_PATH = MODELS / "voices-v1.0.bin"
VOICE = os.environ.get("BOLT_TTS_VOICE", "af_heart")
SPEED = float(os.environ.get("BOLT_TTS_SPEED", "1.15"))

# Strip the things a model writes for the eye but nobody wants read aloud.
_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MARKUP = re.compile(r"[*_#>|]+")
_LINKS = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

_kokoro = None
_lock = threading.Lock()


class TTSUnavailable(RuntimeError):
    pass


def available() -> tuple[bool, str]:
    if not MODEL_PATH.is_file():
        return False, f"kokoro model missing at {MODEL_PATH}"
    if not VOICES_PATH.is_file():
        return False, f"kokoro voices missing at {VOICES_PATH}"
    return True, "ready"


def engine():
    global _kokoro
    with _lock:
        if _kokoro is None:
            ok, reason = available()
            if not ok:
                raise TTSUnavailable(reason)
            from kokoro_onnx import Kokoro
            _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        return _kokoro


def speakable(text: str) -> str:
    """Reduce a model reply to something worth hearing."""
    text = _CODE_BLOCK.sub(" (code omitted) ", text)
    text = _LINKS.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MARKUP.sub("", text)
    return " ".join(text.split()).strip()


def sentences(text: str, max_chars: int = 240) -> Iterator[str]:
    """Split into chunks small enough that the first one starts playing quickly."""
    for sentence in _SENTENCE.split(speakable(text)):
        sentence = sentence.strip()
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars) or max_chars
            yield sentence[:cut].strip()
            sentence = sentence[cut:].strip()
        if sentence:
            yield sentence


def synthesize(text: str, voice: str = VOICE, speed: float = SPEED) -> tuple[np.ndarray, int]:
    samples, sample_rate = engine().create(text, voice=voice, speed=speed, lang="en-us")
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def synthesize_stream(text: str, voice: str = VOICE,
                      speed: float = SPEED) -> Iterator[tuple[np.ndarray, int]]:
    """Yield audio per sentence so playback can start before the rest is generated."""
    for sentence in sentences(text):
        yield synthesize(sentence, voice, speed)

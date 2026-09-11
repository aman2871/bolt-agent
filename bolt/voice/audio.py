"""Microphone capture with energy-based VAD, and playback that yields to speech.

webrtcvad is unusable on modern setuptools (it imports the removed pkg_resources),
so voice activity is detected from short-term energy against a noise floor
measured at startup.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def release_audio_backend() -> None:
    """Fully unload PortAudio.

    Closing a stream is not enough on macOS: PortAudio keeps its CoreAudio host
    layer initialised, the process stays attached to the input device, and the
    system microphone indicator stays lit. Terminating releases it — but nothing
    re-initialises on its own, so acquire_audio_backend() must run before the
    next stream is opened.
    """
    try:
        sd._terminate()
    except Exception:
        pass


def acquire_audio_backend() -> None:
    """Re-initialise PortAudio after a release. Safe to call when already up."""
    try:
        sd._initialize()
    except Exception:
        pass


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resample. Good enough for speech, and avoids a scipy dependency."""
    if source_rate == target_rate or samples.size == 0:
        return samples
    duration = samples.size / source_rate
    target_n = max(1, int(duration * target_rate))
    source_x = np.linspace(0.0, duration, samples.size, endpoint=False)
    target_x = np.linspace(0.0, duration, target_n, endpoint=False)
    return np.interp(target_x, source_x, samples).astype(np.float32)


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))


@dataclass
class VADConfig:
    start_multiple: float = 3.0     # speech starts at this multiple of the noise floor
    end_multiple: float = 1.8       # ...and ends when it drops back below this
    min_speech_ms: int = 300
    silence_ms: int = 800           # trailing silence that ends an utterance
    max_utterance_s: float = 30.0
    floor_min: float = 0.002        # never trust a floor quieter than this


class Microphone:
    """Blocking-ish mic reader that hands out fixed-size frames."""

    def __init__(self, device: int | None = None, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.device = device
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            pass  # overflows are expected under load; dropping a frame is fine
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def __enter__(self) -> "Microphone":
        acquire_audio_backend()
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=FRAME_SAMPLES, device=self.device, callback=self._callback)
        self._stream.start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.drain()
        release_audio_backend()

    def read(self, timeout: float = 1.0) -> np.ndarray | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self) -> None:
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break

    def measure_noise_floor(self, seconds: float = 0.6, warmup: float = 0.3) -> float:
        """Median level of the room. The first frames after opening a device are
        often exact zeros while it spins up, and averaging those in produces a
        floor of 0 that makes every later comparison meaningless."""
        warm_until = time.monotonic() + warmup
        while time.monotonic() < warm_until:
            self.read(timeout=0.5)

        levels = []
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = self.read(timeout=0.5)
            if frame is not None:
                levels.append(rms(frame))
        levels = [x for x in levels if x > 0]
        return float(np.median(levels)) if levels else 0.01


def record_utterance(mic: Microphone, floor: float, config: VADConfig | None = None,
                     stop: threading.Event | None = None) -> np.ndarray | None:
    """Capture one utterance: wait for speech, then record until it stops."""
    config = config or VADConfig()
    floor = max(floor, config.floor_min)
    start_threshold = floor * config.start_multiple
    end_threshold = floor * config.end_multiple

    collected: list[np.ndarray] = []
    speaking = False
    silence_ms = 0
    speech_ms = 0
    started_at = time.monotonic()

    while not (stop and stop.is_set()):
        frame = mic.read(timeout=1.0)
        if frame is None:
            if speaking:
                break
            continue

        level = rms(frame)
        if not speaking:
            if level > start_threshold:
                speaking = True
                collected.append(frame)
                speech_ms = FRAME_MS
            continue

        collected.append(frame)
        if level > end_threshold:
            speech_ms += FRAME_MS
            silence_ms = 0
        else:
            silence_ms += FRAME_MS
            if silence_ms >= config.silence_ms:
                break

        if time.monotonic() - started_at > config.max_utterance_s:
            break

    if not collected or speech_ms < config.min_speech_ms:
        return None
    return np.concatenate(collected)


class Player:
    """Plays audio and stops the moment anything asks it to.

    Writes to its own OutputStream rather than sd.play(). The module-level
    sd.play/sd.get_stream pair refers to one global stream, which is ambiguous
    while the microphone's InputStream is also open — and stopping it is
    all-or-nothing. An explicit stream lets playback be abandoned between small
    blocks, so a stop lands in a few milliseconds.
    """

    # ~21 ms at 24 kHz. Smaller blocks and low latency matter: stream.write()
    # blocks until the device drains, so the block size sets how long a stop
    # takes to be heard.
    BLOCK = 512

    def __init__(self, device: int | None = None):
        self.device = device
        self._interrupt = threading.Event()
        # Consulted between blocks; lets another process stop playback.
        self.should_stop: Callable[[], bool] | None = None

    def interrupt(self) -> None:
        self._interrupt.set()

    def reset(self) -> None:
        """Clear the interrupt. Call once before a reply, never between its
        sentences — an interrupt that arrives while the next sentence is still
        being synthesised must survive until playback can honour it."""
        self._interrupt.clear()

    @property
    def interrupted(self) -> bool:
        return self._interrupt.is_set()

    def play(self, samples: np.ndarray, sample_rate: int) -> bool:
        """Returns True if playback finished, False if it was stopped."""
        if self._interrupt.is_set():
            return False              # interrupted before this sentence began
        samples = np.asarray(samples, dtype="float32").reshape(-1)
        checked_at = time.monotonic()
        try:
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                 device=self.device, blocksize=self.BLOCK) as stream:
                for start in range(0, samples.size, self.BLOCK):
                    if self._interrupt.is_set():
                        return False
                    # An external stop is a file read, so poll it a few times a
                    # second rather than on every block.
                    now = time.monotonic()
                    if self.should_stop and now - checked_at > 0.15:
                        checked_at = now
                        if self.should_stop():
                            return False
                    stream.write(samples[start:start + self.BLOCK])
        except sd.PortAudioError:
            return False
        return True


BARGE_IN_MULTIPLE = float(os.environ.get("BOLT_BARGE_IN_MULTIPLE", "3.0"))


def watch_for_barge_in(mic: Microphone, player: Player, floor: float,
                       stop: threading.Event,
                       multiple: float = BARGE_IN_MULTIPLE) -> threading.Thread:
    """Background watcher that interrupts playback when the user speaks.

    The threshold is deliberately higher than the capture threshold so the
    assistant's own voice leaking into the mic does not cut itself off.
    """
    threshold = max(floor, VADConfig.floor_min) * multiple

    def run() -> None:
        consecutive = 0
        while not stop.is_set():
            frame = mic.read(timeout=0.3)
            if frame is None:
                continue
            consecutive = consecutive + 1 if rms(frame) > threshold else 0
            if consecutive >= 3:  # ~90 ms of sustained speech
                player.interrupt()
                return

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

"""Deciding which frames of a video are worth looking at.

Sampling on a clock is blind: it spends the same budget on a static minute as on
the second something happens. This scans densely and cheaply first — a decimated
pass with a per-frame difference measure — then keeps only the frames where the
picture actually changed.

The idea is lifted from Person-Tracker, which ran YOLO over every Nth frame of a
bus camera and aggregated per batch rather than per frame. The same shape works
here with a much cheaper detector, because we only need "did this change", not
"what is it". `bolt.tools.detect` adds the semantic layer when YOLO is present.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# How densely to scan before choosing. Cheap: decode at low resolution, no model.
# Adaptive: one frame a second is plenty for an hour of CCTV and far too coarse
# for a three-second clip, which would then offer fewer candidates than the
# budget asks for.
# The floor is set by how short an event may be, not by the frame budget: at
# 0.5fps a two-second event falls between scans and is never a candidate. One
# frame a second is the coarsest that still sees a brief flash.
SCAN_FPS_MAX = float(os.environ.get("BOLT_SCAN_FPS_MAX", "8"))
SCAN_FPS_MIN = float(os.environ.get("BOLT_SCAN_FPS_MIN", "1.0"))
# Ceiling on scan frames held in memory, so an hour stays affordable.
MAX_SCAN_FRAMES = int(os.environ.get("BOLT_MAX_SCAN_FRAMES", "3600"))
SCAN_WIDTH = int(os.environ.get("BOLT_SCAN_WIDTH", "160"))
# How far above the footage's own baseline a frame must sit to count as activity,
# in robust deviations. Relative, not absolute: movement in a dark CCTV frame is
# noise in a bright screen recording.
CHANGE_SENSITIVITY = float(os.environ.get("BOLT_CHANGE_SENSITIVITY", "4.0"))


@dataclass
class Moment:
    time: float
    change: float          # 0..1, how different from the previously kept frame
    reason: str            # "change" | "interval" | "start"


def scan_rate(duration: float, budget: int) -> float:
    """Dense enough to catch a brief event, capped so long video stays cheap."""
    if duration <= 0:
        return 1.0
    fps = max(SCAN_FPS_MIN, min(SCAN_FPS_MAX, (budget * 4) / duration))
    if duration * fps > MAX_SCAN_FRAMES:
        fps = MAX_SCAN_FRAMES / duration
    return fps


def _scan_frames(path: Path, fps: float, start: float = 0.0,
                 span: float | None = None) -> list[tuple[float, np.ndarray]]:
    """Small greyscale frames across the whole video, with their timestamps.

    Piped as raw bytes rather than written as images: no temporary files, no PNG
    round-trip, and no image library to install. At 160px wide a frame is a few
    kilobytes, so a full scan of an hour costs little.
    """
    height = max(2, int(SCAN_WIDTH * 9 / 16) // 2 * 2)
    window = (["-ss", f"{start:.3f}"] + (["-t", f"{span:.3f}"] if span else []))
    result = subprocess.run(
        ["ffmpeg", "-v", "error", *window, "-i", str(path),
         "-vf", f"fps={fps:.4f},scale={SCAN_WIDTH}:{height},format=gray",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, timeout=3600)
    if result.returncode != 0 or not result.stdout:
        return []

    size = SCAN_WIDTH * height
    count = len(result.stdout) // size
    buffer = np.frombuffer(result.stdout[:count * size], dtype=np.uint8)
    stack = buffer.reshape(count, height, SCAN_WIDTH)
    return [(start + i / fps, stack[i]) for i in range(count)]


def change_series(frames: list[tuple[float, np.ndarray]]) -> list[tuple[float, float]]:
    """(time, difference from the previous scanned frame) across the whole video."""
    series = []
    for (at, frame), (_, previous) in zip(frames[1:], frames):
        if frame.shape != previous.shape:
            continue
        # int16 so the subtraction cannot wrap around on uint8.
        difference = np.abs(frame.astype(np.int16) - previous.astype(np.int16))
        series.append((at, float(difference.mean()) / 255.0))
    return series


def find_moments(path: Path, budget: int, min_gap: float = 0.0,
                 start: float = 0.0, end: float | None = None) -> list[Moment]:
    """Pick up to `budget` timestamps worth showing a vision model.

    Selection is relative to the footage, not an absolute threshold: what counts
    as movement in a dark CCTV frame is noise in a bright screen recording, so a
    fixed number would need retuning per video. The busiest moments are taken
    first, spaced apart, and any remaining budget covers the quiet stretches so a
    static video is still described rather than ignored.
    """
    from bolt.tools.vision import duration_of
    total = duration_of(path)
    end = end if end is not None else total
    span = max(end - start, 0.001)
    # Scan the window being read, not the whole video: a slice of whole-video
    # moments leaves too few candidates per window to be worth using.
    frames = _scan_frames(path, scan_rate(span, budget), start,
                          span if (start or end < total) else None)
    if not frames:
        return []

    duration = span
    if len(frames) < 3 or budget <= 1:
        return [Moment(frames[0][0], 1.0, "start")]

    series = change_series(frames)
    if not series:
        return [Moment(frames[0][0], 1.0, "start")]

    changes = np.array([c for _, c in series])
    baseline = float(np.median(changes))
    spread = float(np.median(np.abs(changes - baseline))) or 1e-6
    # Robust to outliers, unlike a mean: a few dramatic frames should not raise
    # the bar for everything else.
    floor = baseline + CHANGE_SENSITIVITY * spread

    min_gap = min_gap or max(duration / (budget * 4), 0.5)
    chosen: list[Moment] = [Moment(frames[0][0], 1.0, "start")]

    for at, change in sorted(series, key=lambda pair: pair[1], reverse=True):
        if len(chosen) >= budget:
            break
        if change <= floor:
            break
        if all(abs(at - m.time) >= min_gap for m in chosen):
            chosen.append(Moment(at, change, "change"))

    # Whatever budget is left covers the quiet parts, so nothing goes unseen.
    if len(chosen) < budget:
        step = span / (budget - len(chosen) + 1)
        at = start + step
        while at < end and len(chosen) < budget:
            if all(abs(at - m.time) >= min_gap for m in chosen):
                chosen.append(Moment(at, 0.0, "interval"))
            at += step

    return sorted(chosen, key=lambda m: m.time)

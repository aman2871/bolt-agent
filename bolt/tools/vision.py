"""Vision tools: look at images, and read videos as frames plus their audio.

No model here takes video directly, so a video is reduced to something a vision
model can read: frames sampled across its length (scene changes preferred) and a
transcript of the audio track produced by the whisper model already used for
speech input.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from bolt.core import llm, models, workspace
from bolt.tools import motion
from bolt.core.tools import tool

VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
# 24 frames measured at ~5,300 prompt tokens; 32 at ~7,000. Eight frames across
# two minutes is one every fifteen seconds, which is why it used to describe a
# video rather than follow it.
MAX_FRAMES = int(os.environ.get("BOLT_VIDEO_FRAMES", "24"))
FRAME_WIDTH = int(os.environ.get("BOLT_FRAME_WIDTH", "640"))
SCENE_THRESHOLD = float(os.environ.get("BOLT_SCENE_THRESHOLD", "0.3"))
# Longer videos are read in windows: one pass cannot cover an hour at any
# frame budget, so each window is observed and the observations combined.
WINDOW_SECONDS = float(os.environ.get("BOLT_VIDEO_WINDOW", "120"))
MAX_WINDOWS = int(os.environ.get("BOLT_VIDEO_MAX_WINDOWS", "6"))
FRAMES_PER_WINDOW = int(os.environ.get("BOLT_FRAMES_PER_WINDOW", "16"))
USE_MOTION = os.environ.get("BOLT_MOTION_SAMPLING", "1") != "0"


def _vision_model(ctx=None) -> str:
    """Use the model this run already resolved to, when it can actually see.

    Falls back to the cheapest confirmed vision model — which is what happens
    when a non-vision agent calls these tools directly.
    """
    available = models.vision_models()
    if not available:
        raise RuntimeError("no model here accepts images. Run `bolt models probe`.")
    current = getattr(ctx, "model", "") or ""
    return current if current in available else available[0]


def _ffprobe(path: Path) -> dict:
    if shutil.which("ffprobe") is None:
        return {}
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}


def video_summary(path: Path) -> str:
    info = _ffprobe(path)
    if not info:
        return "(ffprobe unavailable)"
    fmt = info.get("format", {})
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(fmt.get("duration", 0) or 0)
    return (f"{duration:.1f}s, {video.get('width', '?')}x{video.get('height', '?')}, "
            f"{video.get('codec_name', '?')}, audio: {'yes' if audio else 'none'}")


def duration_of(path: Path) -> float:
    info = _ffprobe(path)
    try:
        return float(info.get("format", {}).get("duration", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _timestamps_from(stderr: str) -> list[float]:
    """Presentation times ffmpeg reports for the frames it emitted."""
    return [float(m) for m in re.findall(r"pts_time:([0-9.]+)", stderr)]


def parse_region(region: str) -> tuple[int, int, int, int] | None:
    """`x1,y1,x2,y2` in source pixels, or None.

    Restricting attention to a region is worth more than it sounds: a timestamp
    banner or a status bar changes every second and swamps the change measure,
    so the interesting part of the picture never gets picked.
    """
    if not region:
        return None
    try:
        x1, y1, x2, y2 = (int(part) for part in region.replace(" ", "").split(","))
    except (ValueError, TypeError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _video_filter(region: tuple[int, int, int, int] | None) -> str:
    if region:
        x1, y1, x2, y2 = region
        return f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1},scale={FRAME_WIDTH}:-1"
    return f"scale={FRAME_WIDTH}:-1"


def extract_frames_at(path: Path, out_dir: Path, times: list[float],
                      region: tuple[int, int, int, int] | None = None
                      ) -> list[tuple[Path, float]]:
    """Pull the specific frames named, rather than whatever a clock lands on."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, at in enumerate(times):
        target = out_dir / f"pick_{index:03d}.jpg"
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{at:.3f}", "-i", str(path),
             "-frames:v", "1", "-vf", _video_filter(region), "-y", str(target)],
            capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and target.is_file():
            frames.append((target, at))
    return frames


def extract_frames(path: Path, out_dir: Path, count: int = MAX_FRAMES,
                   start: float = 0.0, end: float | None = None
                   ) -> list[tuple[Path, float]]:
    """Sample frames from a span of video, each with the time it was taken.

    Returns (file, seconds). The timestamp is what lets the model place events in
    time instead of inferring order from a pile of stills.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = duration_of(path)
    end = end if end is not None else total
    span = max(end - start, 0.001)

    window = ["-ss", f"{start:.3f}", "-t", f"{span:.3f}"] if start or end < total else []

    # Scene changes first: they pick the frames a person would pick.
    scene = subprocess.run(
        ["ffmpeg", "-v", "info", *window, "-i", str(path),
         "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',scale={FRAME_WIDTH}:-1,showinfo",
         "-vsync", "vfr", "-frames:v", str(count), str(out_dir / "scene_%03d.jpg")],
        capture_output=True, text=True, timeout=900)
    frames = sorted(out_dir.glob("scene_*.jpg"))
    times = _timestamps_from(scene.stderr)

    # A video with no hard cuts yields no scene changes; sample evenly instead.
    if len(frames) < min(count, 4):
        for stale in frames:
            stale.unlink(missing_ok=True)
        fps = count / span
        even = subprocess.run(
            ["ffmpeg", "-v", "info", *window, "-i", str(path),
             "-vf", f"fps={fps:.6f},scale={FRAME_WIDTH}:-1,showinfo",
             "-frames:v", str(count), str(out_dir / "even_%03d.jpg")],
            capture_output=True, text=True, timeout=900)
        frames = sorted(out_dir.glob("even_*.jpg"))
        times = _timestamps_from(even.stderr)

    paired = []
    for i, frame in enumerate(frames[:count]):
        at = times[i] if i < len(times) else (span * i / max(len(frames) - 1, 1))
        paired.append((frame, start + at))
    return paired


_SEGMENT = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2})\.\d+\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.\d+\]\s*(.+)")


class SpeechUnavailable(RuntimeError):
    """Transcription is not installed — distinct from a video with no speech."""


def extract_audio_segments(path: Path, out_dir: Path) -> list[tuple[float, float, str]]:
    """Speech as (start, end, text), so it can be aligned to the frames it covers.

    One undifferentiated transcript forces the model to guess which words go with
    which picture; timed segments remove the guess.
    """
    from bolt.voice import stt

    ok, reason = stt.available()
    if not ok:
        raise SpeechUnavailable(reason)

    wav = out_dir / "audio.wav"
    extracted = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", "16000", "-f", "wav", str(wav)],
        capture_output=True, text=True, timeout=900)
    if extracted.returncode != 0 or not wav.is_file():
        return []

    result = subprocess.run(
        [stt.BINARY, "-m", str(stt.MODEL), "-f", str(wav),
         "--no-prints", "--language", "en", "--threads", "4"],
        capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        return []

    segments = []
    for line in result.stdout.splitlines():
        match = _SEGMENT.match(line.strip())
        if not match:
            continue
        h1, m1, s1, h2, m2, s2, text = match.groups()
        text = stt._BRACKETED.sub(" ", text).strip()
        if text:
            segments.append((int(h1) * 3600 + int(m1) * 60 + int(s1),
                             int(h2) * 3600 + int(m2) * 60 + int(s2), text))
    return segments


def clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"


def speech_between(segments: list[tuple[float, float, str]],
                   start: float, end: float) -> str:
    covered = [f"[{clock(a)}] {text}" for a, b, text in segments if b >= start and a <= end]
    return "\n".join(covered)


async def _ask_vision(question: str, images: list[Path], context: str = "", ctx=None) -> str:
    model = _vision_model(ctx)
    text = question if not context else f"{context}\n\n{question}"
    message = llm.vision_message(text, images)
    result = await llm.complete(model, [message], temperature=0.1)
    return f"[{model}]\n{result.content}"


def _timeline_message(question: str, header: str,
                      frames: list[tuple[Path, float]],
                      segments: list[tuple[float, float, str]]) -> dict:
    """One message where each frame is labelled with its time and the speech
    around it, so the model reads a timeline rather than a pile of stills."""
    content: list[dict] = [{"type": "text", "text": header}]
    for index, (frame, at) in enumerate(frames, 1):
        previous = frames[index - 2][1] if index > 1 else 0.0
        spoken = speech_between(segments, previous, at)
        label = f"Frame {index} — {clock(at)}"
        if spoken:
            label += f"\nHeard around here: {spoken}"
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url",
                        "image_url": {"url": llm.image_data_url(frame)}})
    content.append({"type": "text", "text": question})
    return {"role": "user", "content": content}


def _pick_frames(path: Path, out_dir: Path, start: float, end: float,
                 wanted: int) -> list[tuple[Path, float]]:
    """Frames chosen by what happens in them, with clock sampling as a fallback.

    Even sampling spends the same budget on a static minute as on the second
    something occurs. A cheap decimated scan finds the moments that differ, so
    the expensive model only sees frames worth seeing.
    """
    if USE_MOTION:
        try:
            moments = motion.find_moments(path, wanted, start=start, end=end)
            times = [m.time for m in moments]
            # Fewer candidates than asked for means the scan could not resolve
            # this video; clock sampling covers it better than a thin selection.
            if len(times) >= max(4, int(wanted * 0.75)):
                picked = extract_frames_at(path, out_dir, times)
                if picked:
                    return picked
        except Exception:
            pass          # any failure here just means falling back to the clock
    return extract_frames(path, out_dir, wanted, start, end)


async def _read_window(path: Path, out_dir: Path, question: str, model: str,
                       segments: list[tuple[float, float, str]],
                       start: float, end: float, total: float,
                       frames_wanted: int) -> tuple[str, int]:
    frames = _pick_frames(path, out_dir, start, end, frames_wanted)
    if not frames:
        return "", 0
    header = (f"These are {len(frames)} frames sampled in order from "
              f"{clock(start)} to {clock(end)} of a {clock(total)} video. "
              f"Each is labelled with the time it was taken.")
    message = _timeline_message(question, header, frames, segments)
    result = await llm.complete(model, [message], temperature=0.1)
    return result.content, len(frames)


@tool(description="Look at an image file and answer a question about it.",
      path="Workspace-relative path to the image.",
      question="What you want to know about the image.")
async def look_at_image(path: str, question: str = "Describe this image in detail.",
                        ctx=None) -> str:
    try:
        target = workspace.resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.is_file():
        return f"ERROR: no file at {path}"
    try:
        return await _ask_vision(question, [target], ctx=ctx)
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


@tool(description=("Watch a video and answer a question about it. Frames are sampled "
                   "across the video with timestamps, the audio is transcribed and "
                   "aligned to them, and long videos are read in windows."),
      path="Workspace-relative path to the video.",
      question="What you want to know about the video.")
async def watch_video(path: str, question: str = "Describe what happens in this video.",
                      ctx=None) -> str:
    try:
        target = workspace.resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.is_file():
        return f"ERROR: no file at {path}"
    if target.suffix.lower() not in VIDEO_TYPES:
        return f"ERROR: {target.suffix} is not a supported video type"
    if shutil.which("ffmpeg") is None:
        return "ERROR: ffmpeg is not installed"

    try:
        model = _vision_model(ctx)
    except RuntimeError as exc:
        return f"ERROR: {exc}"

    with tempfile.TemporaryDirectory(prefix="bolt-video-") as tmp:
        out_dir = Path(tmp)
        total = duration_of(target) or 0.0
        speech_note = ""
        try:
            segments = extract_audio_segments(target, out_dir)
        except SpeechUnavailable as exc:
            segments = []
            speech_note = f" Audio was NOT transcribed ({exc}); this describes the pictures only."

        # One window per WINDOW_SECONDS, capped — an hour of video cannot be read
        # in a single pass at any frame budget.
        windows = 1 if total <= WINDOW_SECONDS else min(
            MAX_WINDOWS, int(total // WINDOW_SECONDS) + 1)
        span = total / windows if windows > 1 else total

        try:
            if windows == 1:
                answer, used = await _read_window(
                    target, out_dir, question, model, segments, 0.0, total or 1.0,
                    total, MAX_FRAMES)
                if not used:
                    return "ERROR: could not extract any frames from this video"
                spoken = (" Audio was transcribed." if segments
                          else speech_note or " No speech found.")
                return (f"[{model}] {used} frames across {clock(total)}.{spoken}\n\n"
                        f"{answer}")

            # Several windows: observe each, then combine.
            # Frames per window, not shared across them: each window is its own
            # request, so the token cost is per call rather than cumulative.
            # Too few and a transition is only bracketed to the sampling gap —
            # 8 frames over 100s locates an event to within ~12s.
            per_window = max(FRAMES_PER_WINDOW, MAX_FRAMES // windows)
            observations = []
            for index in range(windows):
                start = index * span
                end = min(total, start + span)
                observed, used = await _read_window(
                    target, out_dir, "Describe what happens in this part, with times.",
                    model, segments, start, end, total, per_window)
                if used:
                    observations.append(
                        f"## {clock(start)}–{clock(end)}\n{observed}")
                for stale in out_dir.glob("*.jpg"):
                    stale.unlink(missing_ok=True)

            if not observations:
                return "ERROR: could not extract any frames from this video"

            joined = "\n\n".join(observations)
            summary = await llm.complete(model, [{"role": "user", "content":
                f"These are observations from consecutive parts of a {clock(total)} "
                f"video.\n\n{joined}\n\nUsing only these, answer: {question}"}],
                temperature=0.1)
            return (f"[{model}] {windows} windows across {clock(total)}, "
                    f"{per_window} frames each.{speech_note}\n\n{summary.content}")
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"


# Timestamp patterns cameras actually use in filenames. Taken from
# Person-Tracker, which read them off a bus camera's 1-minute dumps.
_FILENAME_TIMES = [
    (re.compile(r"(\d{8})[-_]?(\d{6})"), "%Y%m%d%H%M%S"),
    (re.compile(r"(\d{8})[-_]?(\d{4})(?!\d)"), "%Y%m%d%H%M"),
    (re.compile(r"(\d{4}-\d{2}-\d{2})[T _](\d{2})[-:](\d{2})[-:](\d{2})"), None),
]


def time_from_filename(name: str) -> datetime | None:
    """The moment an image was captured, read from its filename."""
    for pattern, fmt in _FILENAME_TIMES:
        match = pattern.search(name)
        if not match:
            continue
        try:
            if fmt:
                return datetime.strptime("".join(match.groups()), fmt)
            date, hour, minute, second = match.groups()
            return datetime.strptime(f"{date} {hour}:{minute}:{second}",
                                     "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return None


@tool(description=("Read a folder of timestamped images as a sequence over time — "
                   "camera dumps, screenshot series, exported frames. Times come "
                   "from the filenames, falling back to file modification time."),
      path="Workspace-relative folder of images.",
      question="What you want to know about the sequence.",
      region="Optional 'x1,y1,x2,y2' crop, to ignore banners or watch one panel.")
async def watch_image_sequence(path: str,
                               question: str = "What changes across these images?",
                               region: str = "", ctx=None) -> str:
    try:
        folder = workspace.resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not folder.is_dir():
        return f"ERROR: {path} is not a directory"

    images = [p for p in sorted(folder.iterdir())
              if p.suffix.lower() in llm.IMAGE_TYPES]
    if not images:
        return f"no images found in {path}"

    # Filename time first, modification time as a fallback: a camera writing
    # every minute encodes the moment in the name, and sorting by name alone
    # goes wrong the moment the numbering wraps.
    stamped = []
    for image in images:
        at = time_from_filename(image.name)
        stamped.append((image, at or datetime.fromtimestamp(image.stat().st_mtime)))
    stamped.sort(key=lambda pair: pair[1])

    try:
        model = _vision_model(ctx)
    except RuntimeError as exc:
        return f"ERROR: {exc}"

    total = len(stamped)
    # Spread across the whole sequence. Taking the first N instead silently
    # truncates the timeline — the end of the sequence is usually the point.
    if total <= MAX_FRAMES:
        chosen = stamped
    else:
        picks = {round(i * (total - 1) / (MAX_FRAMES - 1)) for i in range(MAX_FRAMES)}
        chosen = [stamped[i] for i in sorted(picks)]
    first, last = stamped[0][1], stamped[-1][1]

    crop = parse_region(region)
    with tempfile.TemporaryDirectory(prefix="bolt-seq-") as tmp:
        prepared = []
        for index, (image, at) in enumerate(chosen):
            target = Path(tmp) / f"seq_{index:03d}.jpg"
            result = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(image),
                 "-vf", _video_filter(crop), "-y", str(target)],
                capture_output=True, text=True, timeout=120)
            prepared.append((target if result.returncode == 0 else image, at))

        content: list[dict] = [{"type": "text", "text":
            f"These are {len(prepared)} images from a sequence of {total}, in order, "
            f"spanning {first:%Y-%m-%d %H:%M:%S} to {last:%Y-%m-%d %H:%M:%S}. "
            f"Each is labelled with when it was captured."}]
        for image, at in prepared:
            content.append({"type": "text", "text": f"{at:%H:%M:%S}"})
            content.append({"type": "image_url",
                            "image_url": {"url": llm.image_data_url(image)}})
        content.append({"type": "text", "text": question})

        try:
            result = await llm.complete(model, [{"role": "user", "content": content}],
                                        temperature=0.1)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    sampled = "" if len(prepared) == total else " spread evenly"
    return (f"[{model}] {len(prepared)} of {total} images{sampled}, "
            f"{first:%H:%M:%S}–{last:%H:%M:%S}\n\n{result.content}")


@tool(description="List the images and videos available in a workspace directory.",
      path="Workspace-relative directory to look in.")
def list_media(path: str = ".") -> str:
    try:
        target = workspace.resolve(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if not target.is_dir():
        return f"ERROR: {path} is not a directory"
    media = [p for p in sorted(target.rglob("*"))
             if p.suffix.lower() in (llm.IMAGE_TYPES | VIDEO_TYPES)]
    if not media:
        return "no images or videos found"
    return "\n".join(f"{p.relative_to(workspace.root())}  "
                     f"{p.stat().st_size // 1024}KB" for p in media[:200])

"""The always-on voice loop: listen, route, run, speak."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import bolt.registry  # noqa: F401  (registers tools)
from bolt.core import roster
from bolt.core.loop import run_agent
from bolt.voice import audio, control, stt, tts, wake

Reporter = Callable[[str, dict[str, Any]], None]


@dataclass
class VoiceConfig:
    default_agent: str = "manager"
    budget_tokens: int = 200_000
    input_device: int | None = None
    output_device: int | None = None
    speak_replies: bool = True
    max_spoken_chars: int = 900
    # Listen only while the dashboard is open in a browser tab. Off means listen
    # always, which is what `bolt listen --always` gives you.
    require_tab: bool = True


def preflight() -> list[str]:
    """Problems that would stop a voice session, as human-readable lines."""
    problems = []
    ok, reason = stt.available()
    if not ok:
        problems.append(f"speech to text: {reason}")
    ok, reason = tts.available()
    if not ok:
        problems.append(f"text to speech: {reason}")
    try:
        import sounddevice as sd
        if not any(d["max_input_channels"] > 0 for d in sd.query_devices()):
            problems.append("no input device found")
    except Exception as exc:
        problems.append(f"audio devices unavailable: {exc}")
    return problems


async def handle_utterance(text: str, config: VoiceConfig,
                           report: Reporter, on_event=None) -> tuple[str, str] | None:
    """Route one transcript to an agent and return (agent, reply)."""
    address = wake.resolve(text, config.default_agent)
    if not address.addressed:
        report("ignored", {"text": text})
        return None
    if not address.text:
        return address.agent, "I'm listening."

    report("routed", {"agent": address.agent, "text": address.text,
                      "matched": address.matched})
    spec = roster.get(address.agent)
    result = await run_agent(spec, address.text, budget_tokens=config.budget_tokens,
                             extra={"on_event": on_event}, on_event=on_event)
    reply = result.content if result.ok else f"That failed. {result.error}"
    return address.agent, reply


def speak(player: audio.Player, mic: audio.Microphone, floor: float,
          text: str, config: VoiceConfig, report: Reporter) -> None:
    """Speak a reply, stopping if the user talks over it or anything asks it to."""
    spoken = tts.speakable(text)[: config.max_spoken_chars]
    if not spoken:
        return

    # Anything newer than this marker is a stop meant for this reply.
    marker = control.stop_seq()
    player.reset()
    player.should_stop = lambda: control.stop_requested_since(marker)

    stop_watch = threading.Event()
    watcher = audio.watch_for_barge_in(mic, player, floor, stop_watch)
    interrupted = False
    control.mark_speaking(True)
    try:
        for samples, rate in tts.synthesize_stream(spoken):
            # Check before synthesising the next sentence too, so an interruption
            # during generation is not paid for with more speech.
            if player.interrupted or control.stop_requested_since(marker):
                interrupted = True
                report("barge_in", {"by": "you" if player.interrupted else "stop request"})
                break
            if not player.play(samples, rate):
                interrupted = True
                report("barge_in", {"by": "you" if player.interrupted else "stop request"})
                break
    finally:
        control.mark_speaking(False)
        player.should_stop = None
        stop_watch.set()
        watcher.join(timeout=1.0)
        # Only discard buffered audio when the reply ran to completion — that is
        # the assistant's own voice leaking back in. After a barge-in the buffer
        # holds what the user just said, and throwing it away would force them to
        # repeat the command they interrupted with.
        if not interrupted:
            mic.drain()


async def run_session(config: VoiceConfig, report: Reporter, on_event=None) -> None:
    """Listen and respond until interrupted."""
    problems = preflight()
    if problems:
        for problem in problems:
            report("problem", {"detail": problem})
        return

    loop = asyncio.get_running_loop()
    player = audio.Player(device=config.output_device)
    report("ready", {"noise_floor": None, "agents": ", ".join(roster.names())})

    # The microphone is opened only while we are actually allowed to listen, and
    # closed the moment we are not. Merely ignoring the frames is not enough: an
    # open InputStream is a live capture, so macOS shows the orange recording
    # indicator and the process genuinely holds the microphone.
    mic: audio.Microphone | None = None
    floor = 0.01
    gated = False

    def close_mic() -> None:
        nonlocal mic
        if mic is not None:
            mic.__exit__(None, None, None)
            mic = None
            report("mic_released", {})

    def open_mic() -> float:
        nonlocal mic
        mic = audio.Microphone(device=config.input_device)
        mic.__enter__()
        measured = mic.measure_noise_floor()
        report("mic_open", {"noise_floor": round(measured, 5)})
        return measured

    def gate_reason() -> str:
        if not control.is_enabled():
            return "voice is switched off in settings"
        if config.require_tab and not control.tab_open():
            return "no dashboard tab open"
        if control.is_paused():
            return "paused while the dashboard mic is recording"
        return ""

    def should_gate() -> bool:
        return bool(gate_reason())

    try:
        while True:
            reason = gate_reason()

            if reason:
                close_mic()                      # releases the OS microphone
                if not gated:
                    report("gated", {"reason": reason})
                    gated = True
                await asyncio.sleep(0.5)
                continue

            if mic is None:
                floor = await loop.run_in_executor(None, open_mic)
                gated = False
                report("ungated", {})
            elif gated:
                gated = False
                mic.drain()

            report("listening", {})
            current = mic
            # record_utterance blocks until somebody speaks. Without a way to
            # abort it, a tab closing during silence would not be noticed and the
            # microphone would stay open indefinitely.
            abort = threading.Event()
            capture = loop.run_in_executor(
                None, lambda: audio.record_utterance(current, floor, stop=abort))
            while not capture.done():
                await asyncio.sleep(0.3)
                if should_gate():
                    abort.set()
            samples = await capture
            if samples is None:
                continue
            if abort.is_set():
                continue          # gated mid-capture; drop it and release the mic

            report("heard", {"seconds": round(samples.size / audio.SAMPLE_RATE, 2)})
            text = await loop.run_in_executor(
                None, lambda: stt.transcribe(np.asarray(samples), audio.SAMPLE_RATE))
            if not text:
                continue
            report("transcript", {"text": text})

            outcome = await handle_utterance(text, config, report, on_event)
            if outcome is None:
                continue
            agent, reply = outcome
            report("reply", {"agent": agent, "text": reply})

            if config.speak_replies and mic is not None:
                speaking_mic = mic
                await loop.run_in_executor(
                    None, lambda: speak(player, speaking_mic, floor, reply, config, report))
    finally:
        close_mic()

"""Ollama client speaking the OpenAI-compatible chat completions API."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import random
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

BASE_URL = os.environ.get("BOLT_OLLAMA_URL", "http://localhost:11434/v1")
API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")


def endpoint() -> tuple[str, str]:
    """Where to call and with what key.

    Settings saved in the dashboard win over the environment, so a person who
    installs Bolt can paste their Ollama key into the UI instead of editing a
    compose file. The environment stays the fallback, which is what a server
    deployment wants.
    """
    try:
        from bolt.core import store
        url = store.get_setting("ollama.base_url") or BASE_URL
        key = store.get_setting("ollama.api_key") or API_KEY
    except Exception:            # the store may not exist yet during setup
        url, key = BASE_URL, API_KEY
    return url, key
REQUEST_TIMEOUT = float(os.environ.get("BOLT_LLM_TIMEOUT", "180"))
MAX_ATTEMPTS = int(os.environ.get("BOLT_LLM_RETRIES", "5"))

# Switching networks (office wifi to home wifi, sleep/wake, VPN up or down) leaves
# pooled sockets half-open: the peer is unreachable but the OS keeps the socket in
# ESTABLISHED, so a read on it blocks until TCP keepalive gives up, which by
# default takes hours. Expiring idle connections quickly means a request after a
# network change opens a fresh socket instead of inheriting a dead one.
KEEPALIVE_EXPIRY = float(os.environ.get("BOLT_LLM_KEEPALIVE", "30"))
CONNECT_TIMEOUT = float(os.environ.get("BOLT_LLM_CONNECT_TIMEOUT", "10"))

# Every model is cloud-hosted, so cap how many requests are in flight at once.
_semaphore = asyncio.Semaphore(int(os.environ.get("BOLT_LLM_CONCURRENCY", "4")))

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# --- effort ---------------------------------------------------------------
# Ollama's OpenAI-compatible endpoint accepts `think` and `reasoning_effort` and
# then ignores them: measured, `think: false` still produced 155 characters of
# reasoning there, against 0 on the native API. So an agent that asks for an
# effort level is routed to /api/chat instead, which honours it — on kimi-k3,
# 10 output tokens at `false` against 175 at `true` for the same right answer.
EFFORT_THINK: dict[str, Any] = {
    "off": False, "low": "low", "medium": "medium", "high": "high", "max": True,
}
EFFORT_CHOICES = ["default", *EFFORT_THINK]


def think_for(effort: str | None) -> Any:
    """The `think` value for an effort name, or None to stay on the usual path."""
    return EFFORT_THINK.get((effort or "default").strip().lower())


async def probe_completion(base_url: str, key: str | None, model: str) -> None:
    """The smallest real request there is, against an endpoint we do not yet use.

    Listing models proves nothing: ollama.com serves its catalogue publicly and
    then refuses every completion. Raises on failure.
    """
    headers = {"Content-Type": "application/json"}
    if key and key != "ollama":
        headers["Authorization"] = f"Bearer {key}"
    async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=CONNECT_TIMEOUT)) as probe:
        response = await probe.post(
            f"{base_url.rstrip('/')}/chat/completions", headers=headers,
            json={"model": model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}]})
    if response.status_code != 200:
        raise LLMError(f"HTTP {response.status_code}: {response.text[:200]}")


def native_url(base: str) -> str:
    """/api/chat beside the OpenAI-compatible /v1 the rest of Bolt talks to."""
    trimmed = base.rstrip("/")
    if trimmed.endswith("/v1"):
        trimmed = trimmed[:-3]
    return trimmed.rstrip("/") + "/api/chat"


def _native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat messages in the shape /api/chat expects.

    Three things differ: images ride as a list of bare base64 beside the text
    rather than inside the content array, tool-call arguments are an object and
    not a JSON string, and a tool result is matched to its call by name.
    """
    out = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        converted: dict[str, Any] = {"role": role}
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text") or "")
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url") or ""
                    if url.startswith("data:"):
                        images.append(url.split(",", 1)[-1])
            converted["content"] = "\n".join(texts)
            if images:
                converted["images"] = images
        else:
            converted["content"] = content or ""

        if role == "tool" and message.get("name"):
            converted["tool_name"] = message["name"]

        calls = message.get("tool_calls")
        if calls:
            native_calls = []
            for call in calls:
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                native_calls.append(
                    {"function": {"name": fn.get("name") or "", "arguments": args or {}}})
            converted["tool_calls"] = native_calls
        out.append(converted)
    return out


def _native_tool_calls(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Native tool calls back in OpenAI shape. They carry no id, so mint one."""
    out = []
    for index, call in enumerate(calls or []):
        fn = call.get("function") or {}
        out.append({
            "id": f"call_{index}_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": fn.get("name") or "",
                         "arguments": json.dumps(fn.get("arguments") or {})},
        })
    return out


def _native_payload(model: str, messages: list[dict[str, Any]],
                    tools: list[dict[str, Any]] | None, temperature: float,
                    max_tokens: int | None, think: Any, stream: bool) -> dict[str, Any]:
    options: dict[str, Any] = {"temperature": temperature}
    if max_tokens:
        options["num_predict"] = max_tokens
    payload: dict[str, Any] = {
        "model": model,
        "messages": _native_messages(messages),
        "think": think,
        "stream": stream,
        "options": options,
    }
    if tools:
        payload["tools"] = tools
    return payload


class LLMError(RuntimeError):
    """Raised when a completion could not be obtained after all retries."""


@dataclass
class Completion:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    attempts: int = 1
    model: str = ""
    raw_message: dict[str, Any] = field(default_factory=dict)


_client: httpx.AsyncClient | None = None


_client_for: tuple[str, str] | None = None


def client() -> httpx.AsyncClient:
    global _client, _client_for
    url, key = endpoint()
    # A key or URL changed in settings has to take effect without a restart, and
    # a pooled client still holds the old Authorization header.
    if _client is not None and _client_for != (url, key):
        old, _client, _client_for = _client, None, None
        asyncio.get_event_loop().create_task(old.aclose())
    if _client is None:
        _client_for = (url, key)
        _client = httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
            headers={"Authorization": f"Bearer {key}"},
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=32,
                                keepalive_expiry=KEEPALIVE_EXPIRY),
        )
    return _client


async def reset_client() -> None:
    """Drop every pooled connection. Called when a request fails at transport
    level, so the retry does not reuse the socket that just died."""
    await aclose()


async def aclose() -> None:
    global _client, _client_for
    if _client is not None:
        await _client.aclose()
        _client = None
        _client_for = None


def _backoff(attempt: int) -> float:
    return min(30.0, 1.5 * (2 ** (attempt - 1))) * (0.7 + 0.6 * random.random())


async def complete(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
    effort: str | None = None,
) -> Completion:
    """One chat completion, retrying transient failures with exponential backoff."""
    think = think_for(effort)
    if think is not None:
        return await _complete_native(model, messages, tools, temperature,
                                      max_tokens, think)
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format

    started = time.monotonic()
    last_error: str = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with _semaphore:
                resp = await client().post("/chat/completions", json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            # The pool may be holding sockets stranded by a network change.
            await reset_client()
        else:
            if resp.status_code == 200:
                return _parse(resp.json(), model, started, attempt)
            last_error = f"HTTP {resp.status_code}: {resp.text[:400]}"
            if resp.status_code not in RETRY_STATUS:
                raise LLMError(f"{model} call failed, {last_error}")

        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(_backoff(attempt))

    raise LLMError(f"{model} call failed after {MAX_ATTEMPTS} attempts, last error: {last_error}")


async def _complete_native(model: str, messages: list[dict[str, Any]],
                           tools: list[dict[str, Any]] | None, temperature: float,
                           max_tokens: int | None, think: Any) -> Completion:
    """The same call against Ollama's own API, which honours `think`."""
    url = native_url(endpoint()[0])
    payload = _native_payload(model, messages, tools, temperature, max_tokens,
                              think, stream=False)
    started = time.monotonic()
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with _semaphore:
                resp = await client().post(url, json=payload)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            await reset_client()
        else:
            if resp.status_code == 200:
                data = resp.json()
                message = data.get("message") or {}
                return Completion(
                    content=message.get("content") or "",
                    tool_calls=_native_tool_calls(message.get("tool_calls")),
                    finish_reason=data.get("done_reason") or "stop",
                    prompt_tokens=int(data.get("prompt_eval_count") or 0),
                    completion_tokens=int(data.get("eval_count") or 0),
                    latency_ms=int((time.monotonic() - started) * 1000),
                    attempts=attempt, model=model, raw_message=message)
            last_error = f"HTTP {resp.status_code}: {resp.text[:400]}"
            if resp.status_code not in RETRY_STATUS:
                raise LLMError(f"{model} call failed, {last_error}")
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(_backoff(attempt))

    raise LLMError(f"{model} call failed after {MAX_ATTEMPTS} attempts, "
                   f"last error: {last_error}")


def _parse(data: dict[str, Any], model: str, started: float, attempts: int) -> Completion:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = data.get("usage") or {}
    raw_calls = message.get("tool_calls") or []

    calls = []
    for i, call in enumerate(raw_calls):
        fn = call.get("function") or {}
        calls.append({
            "id": call.get("id") or f"call_{i}",
            "type": "function",
            "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments") or "{}"},
        })

    return Completion(
        content=message.get("content") or "",
        tool_calls=calls,
        finish_reason=choice.get("finish_reason") or "stop",
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        latency_ms=int((time.monotonic() - started) * 1000),
        attempts=attempts,
        model=model,
        raw_message=message,
    )


# --- multimodal helpers -------------------------------------------------------

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_IMAGE_BYTES = 6_000_000


def user_image(path: str) -> str:
    """Data URL for an image path a person supplied.

    A relative path means the workspace, not this process's working directory —
    inside the container that is /app, where nothing the user names exists. It
    goes through the same resolver as every other file, so an image cannot be
    used to read outside the allowed roots. Frames Bolt extracts itself are
    already absolute temp paths and use image_data_url directly.
    """
    if str(path).startswith("data:"):
        return str(path)
    from bolt.core import workspace
    return image_data_url(workspace.resolve(str(path)))


def image_data_url(path: Path | str) -> str:
    """Read an image file into a data: URL suitable for an image_url content block."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no image at {path}")
    if path.suffix.lower() not in IMAGE_TYPES:
        raise ValueError(f"{path.suffix} is not a supported image type")
    # Size first: reading the file to discover it is too big costs exactly as
    # much memory as the file, which is the thing the limit exists to prevent.
    size = path.stat().st_size
    if size > MAX_IMAGE_BYTES:
        raise ValueError(f"{path.name} is {size // 1_000_000}MB, over the "
                         f"{MAX_IMAGE_BYTES // 1_000_000}MB limit")
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def vision_message(text: str, images: list[Path | str], role: str = "user") -> dict[str, Any]:
    """Build one chat message carrying text plus one or more images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        url = image if str(image).startswith("data:") else image_data_url(image)
        content.append({"type": "image_url", "image_url": {"url": url}})
    return {"role": role, "content": content}

async def _stream_native(model: str, messages: list[dict[str, Any]],
                         tools: list[dict[str, Any]] | None, temperature: float,
                         max_tokens: int | None, think: Any,
                         on_text: Callable[[str], None] | None,
                         on_reasoning: Callable[[str], None] | None) -> Completion:
    """Streamed /api/chat: newline-delimited JSON, one object per fragment."""
    url = native_url(endpoint()[0])
    payload = _native_payload(model, messages, tools, temperature, max_tokens,
                              think, stream=True)
    started = time.monotonic()
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        text, calls = [], []
        prompt_tokens = completion_tokens = 0
        finish = "stop"
        try:
            async with _semaphore:
                async with client().stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        body = (await response.aread()).decode(errors="replace")
                        last_error = f"HTTP {response.status_code}: {body[:400]}"
                        if response.status_code not in RETRY_STATUS:
                            raise LLMError(f"{model} call failed, {last_error}")
                        raise _Retry()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        message = chunk.get("message") or {}
                        if piece := message.get("content"):
                            text.append(piece)
                            if on_text:
                                on_text(piece)
                        if thought := message.get("thinking"):
                            if on_reasoning:
                                on_reasoning(thought)
                        if message.get("tool_calls"):
                            calls.extend(message["tool_calls"])
                        if chunk.get("done"):
                            finish = chunk.get("done_reason") or "stop"
                            prompt_tokens = int(chunk.get("prompt_eval_count") or 0)
                            completion_tokens = int(chunk.get("eval_count") or 0)
        except _Retry:
            pass
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            await reset_client()
        else:
            return Completion(
                content="".join(text), tool_calls=_native_tool_calls(calls),
                finish_reason=finish, prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=int((time.monotonic() - started) * 1000),
                attempts=attempt, model=model)
        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(_backoff(attempt))

    raise LLMError(f"{model} stream failed after {MAX_ATTEMPTS} attempts, "
                   f"last error: {last_error}")


class _Retry(Exception):
    """A retryable status inside the streaming context manager."""


async def stream(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    on_text: Callable[[str], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    effort: str | None = None,
) -> Completion:
    """Same contract as complete(), but text arrives as it is generated.

    `on_text` receives visible answer fragments; `on_reasoning` receives the
    model's thinking, which these models emit as a separate field and which is
    worth showing as progress rather than as the answer.

    Retries restart the stream from the beginning. A half-finished answer is not
    something we can resume, and replaying is better than returning a truncated
    one — callers are told to discard what they have via `on_reset`.
    """
    think = think_for(effort)
    if think is not None:
        return await _stream_native(model, messages, tools, temperature,
                                    max_tokens, think, on_text, on_reasoning)
    payload: dict[str, Any] = {
        "model": model, "messages": messages,
        "temperature": temperature, "stream": True,
        # Without this a streamed response reports no usage at all, which would
        # silently zero the token and cost figures the dashboard runs on.
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = tools
    if max_tokens:
        payload["max_tokens"] = max_tokens

    started = time.monotonic()
    last_error = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        content: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        usage: dict[str, Any] = {}
        try:
            async with _semaphore:
                async with client().stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code != 200:
                        raw = (await response.aread()).decode("utf-8", "replace")
                        last_error = f"HTTP {response.status_code}: {raw[:400]}"
                        if response.status_code not in RETRY_STATUS:
                            raise LLMError(f"{model} call failed, {last_error}")
                        raise _Retry()

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        body = line[5:].strip()
                        if body == "[DONE]":
                            break
                        try:
                            event = json.loads(body)
                        except json.JSONDecodeError:
                            continue

                        usage = event.get("usage") or usage
                        choice = (event.get("choices") or [{}])[0]
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}

                        if piece := delta.get("content"):
                            content.append(piece)
                            if on_text:
                                on_text(piece)
                        if thought := delta.get("reasoning"):
                            if on_reasoning:
                                on_reasoning(thought)

                        # Arguments can arrive in fragments, keyed by index.
                        for call in delta.get("tool_calls") or []:
                            index = call.get("index", 0)
                            slot = calls.setdefault(
                                index, {"id": call.get("id") or f"call_{index}",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}})
                            if call.get("id"):
                                slot["id"] = call["id"]
                            fn = call.get("function") or {}
                            if fn.get("name"):
                                slot["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                slot["function"]["arguments"] += fn["arguments"]

        except _Retry:
            pass
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            await reset_client()
        else:
            text = "".join(content)
            return Completion(
                content=text,
                tool_calls=[calls[i] for i in sorted(calls)],
                finish_reason=finish_reason,
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
                latency_ms=int((time.monotonic() - started) * 1000),
                attempts=attempt,
                model=model,
                raw_message={"role": "assistant", "content": text},
            )

        if attempt < MAX_ATTEMPTS:
            await asyncio.sleep(_backoff(attempt))

    raise LLMError(f"{model} stream failed after {MAX_ATTEMPTS} attempts, "
                   f"last error: {last_error}")


class _Retry(Exception):
    """Internal signal that a streamed attempt should be retried."""


def estimate_tokens(text: str) -> int:
    """Rough count for providers that omit usage on streamed responses."""
    return max(1, len(text) // 4)

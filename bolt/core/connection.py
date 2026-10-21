"""Where Bolt gets its models, and whether that actually works.

Bolt talks to one endpoint at a time, but a person does not think in
endpoints — they think "my local models" and "the cloud ones". Those are not
the either/or the single URL box implied: the Ollama app on your machine
serves what you have pulled *and* proxies the cloud models when it is signed
in, which is one connection covering both and needing no API key.

So the choice here is a source, not a URL, and every source reports what it
can actually do rather than what it claims.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from bolt.core import llm

# Inside a container the Ollama app on the host is reachable by this name; on
# bare metal it is plain localhost. Both are tried, nearest first.
LOCAL_URLS = [
    os.environ.get("BOLT_OLLAMA_URL", "http://host.docker.internal:11434/v1"),
    "http://host.docker.internal:11434/v1",
    "http://localhost:11434/v1",
]
CLOUD_URL = "https://ollama.com/v1"
PROBE_TIMEOUT = 6.0


@dataclass
class Source:
    id: str                       # local | cloud | custom
    name: str
    base_url: str
    reachable: bool = False
    models: list[str] = field(default_factory=list)
    needs_key: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "base_url": self.base_url,
                "reachable": self.reachable, "models": self.models,
                "model_count": len(self.models), "needs_key": self.needs_key,
                "cloud_models": sum(1 for m in self.models if m.endswith(":cloud")),
                "detail": self.detail}


async def _list_models(url: str, key: str | None = None) -> list[str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
        response = await client.get(f"{url.rstrip('/')}/models", headers=headers)
        response.raise_for_status()
        return sorted(m.get("id", "") for m in (response.json().get("data") or [])
                      if m.get("id"))


async def probe_local() -> Source:
    """The Ollama app, wherever it is reachable from here."""
    source = Source("local", "Ollama on this machine", LOCAL_URLS[0])
    seen = []
    for url in LOCAL_URLS:
        if url in seen:
            continue
        seen.append(url)
        try:
            models = await _list_models(url)
        except Exception:
            continue
        source.base_url, source.reachable, source.models = url, True, models
        cloud = [m for m in models if m.endswith(":cloud")]
        # Cloud entries only appear once the app is signed in, so their
        # presence is the signal — there is nothing else to ask.
        source.detail = (
            f"{len(models)} models, {len(cloud)} of them cloud — signed in, "
            f"so cloud models work without an API key here."
            if cloud else
            f"{len(models)} local models. Sign in to the Ollama app to reach "
            f"the cloud ones too, or add a key below.")
        return source
    source.detail = ("Not running, or not reachable from Bolt. Start the Ollama "
                     "app, or use a key below.")
    return source


async def probe_cloud(key: str | None) -> Source:
    """ollama.com. It lists its catalogue to anyone and answers only with a key."""
    source = Source("cloud", "Ollama Cloud", CLOUD_URL, needs_key=True)
    try:
        source.models = await _list_models(CLOUD_URL, key)
        source.reachable = True
    except Exception as exc:
        source.detail = f"Could not reach ollama.com ({type(exc).__name__})."
        return source

    if not key:
        # The listing is public, which is exactly why it cannot be the check.
        source.detail = (f"{len(source.models)} models listed, but ollama.com "
                         f"answers nothing without an API key.")
        return source

    try:
        await llm.probe_completion(CLOUD_URL, key, source.models[0])
    except Exception as exc:
        source.detail = f"Key rejected: {str(exc)[:120]}"
        return source
    source.detail = f"Key accepted — {len(source.models)} models."
    source.needs_key = False
    return source


def mode_for(url: str) -> str:
    bare = (url or "").rstrip("/")
    if bare.startswith(CLOUD_URL.rstrip("/")):
        return "cloud"
    if any(bare == u.rstrip("/") for u in LOCAL_URLS) or ":11434" in bare:
        return "local"
    return "custom"

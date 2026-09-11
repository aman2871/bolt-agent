"""Tool registry: decorate a function, get an OpenAI-style schema and a dispatcher."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable

_PY_TO_JSON = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", dict: "object"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    @property
    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


REGISTRY: dict[str, Tool] = {}


def _infer_parameters(fn: Callable[..., Any], described: dict[str, str]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name == "ctx":
            continue
        json_type = _PY_TO_JSON.get(param.annotation, "string")
        props[name] = {"type": json_type, "description": described.get(name, "")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def tool(name: str | None = None, description: str = "", **param_docs: str):
    """Register a function as a tool. A `ctx` first parameter is injected, not exposed."""
    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__

        # `tool(name=...)` names the tool, so a parameter called `name` silently
        # steals it — the tool then registers under its own help text. Checking
        # the docs against the signature turns that into an immediate error.
        parameters = set(inspect.signature(fn).parameters) - {"ctx"}
        unknown = set(param_docs) - parameters
        if unknown:
            raise TypeError(
                f"{fn.__name__}: documented parameters {sorted(unknown)} do not exist "
                f"(has {sorted(parameters)}). Note that `name=` names the tool itself.")
        REGISTRY[tool_name] = Tool(
            name=tool_name,
            description=description or (fn.__doc__ or "").strip(),
            parameters=_infer_parameters(fn, param_docs),
            fn=fn,
        )
        return fn
    return wrap


def schemas(names: list[str]) -> list[dict[str, Any]]:
    """Schemas for the named tools, skipping any that are not registered."""
    return [REGISTRY[n].schema for n in names if n in REGISTRY]


def wants_ctx(fn: Callable[..., Any]) -> bool:
    return "ctx" in inspect.signature(fn).parameters


async def dispatch(name: str, arguments: str | dict[str, Any], ctx: Any = None) -> str:
    """Run a tool by name. Always returns a string — errors come back as text for the model."""
    entry = REGISTRY.get(name)
    if entry is None:
        return f"ERROR: unknown tool {name!r}. Available: {', '.join(sorted(REGISTRY))}"

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments for {name} are not valid JSON ({exc})"
    else:
        args = dict(arguments or {})

    if not isinstance(args, dict):
        return f"ERROR: arguments for {name} must be a JSON object"

    if wants_ctx(entry.fn):
        args["ctx"] = ctx

    try:
        result = entry.fn(**args)
        if inspect.isawaitable(result):
            result = await result
    except TypeError as exc:
        return f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:
        return f"ERROR: {name} raised {type(exc).__name__}: {exc}"

    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):
        return str(result)

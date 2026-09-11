"""Deciding who a spoken utterance is addressed to.

Custom wake words for six agent names would each need their own trained model, so
addressing works from the transcript instead: an utterance is for Bolt when it
opens with the system name or an agent's name, and that name picks the agent.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass

from bolt.core import roster

# "bolt" plus what whisper actually returns for it. Measured, not guessed: under
# noise it comes back as "bold" most often, and the rest were observed in testing.
# A consistently-misheard name is fine as long as the mishearings are listed.
SYSTEM_NAMES = {"bolt", "bold", "volt", "boult", "bolte", "holt", "boldt"}
GREETINGS = ("hey", "hay", "hi", "ok", "okay", "a")
# Whisper often glues the greeting to the name: "Hey Bolt" comes back "Hayhive".
_GLUED = re.compile(r"^(?:" + "|".join(GREETINGS) + r")(?P<name>[a-z]+)$")
FUZZY_CUTOFF = 0.82

# Names whisper tends to mangle, mapped back to the agent they mean.
ALIASES: dict[str, str] = {
    "forge": "forge", "ford": "forge", "forged": "forge", "george": "forge",
    "scout": "scout", "scout's": "scout", "skout": "scout",
    "sentinel": "sentinel", "sentinal": "sentinel", "centinel": "sentinel",
    "archivist": "archivist", "archive": "archivist", "archivest": "archivist",
    "ops": "ops", "op": "ops", "opps": "ops",
    "manager": "manager", "management": "manager",
}

REQUIRE_ADDRESS = os.environ.get("BOLT_REQUIRE_ADDRESS", "1") != "0"

# Filler that follows the name in natural speech — "Bolt bro, let's..." — and
# should not end up in the prompt the agent receives.
FILLER = {"bro", "buddy", "mate", "man", "dude", "please", "pls", "ok", "okay",
          "so", "um", "uh", "hey", "yo"}
_WORD = re.compile(r"[a-z']+")


@dataclass
class Address:
    agent: str
    text: str
    addressed: bool
    matched: str | None = None


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _close(word: str, options: set[str]) -> str | None:
    """Nearest option to a word, for transcripts that are close but not exact."""
    if word in options:
        return word
    hits = difflib.get_close_matches(word, options, n=1, cutoff=FUZZY_CUTOFF)
    return hits[0] if hits else None


def _as_system_name(word: str) -> bool:
    return _close(word, SYSTEM_NAMES) is not None


def _as_agent(word: str, known: set[str]) -> str | None:
    candidate = ALIASES.get(word)
    if candidate in known:
        return candidate
    return _close(word, known)


def resolve(utterance: str, default_agent: str = "manager") -> Address:
    """Work out which agent an utterance is for, and strip the address from it."""
    text = utterance.strip()
    words = _words(text)
    if not words:
        return Address(default_agent, "", False)

    known = set(roster.names())
    consumed = 0
    addressed = False
    agent = default_agent
    matched: str | None = None

    # A greeting may be glued onto the name that follows it.
    glued = _GLUED.match(words[0])
    if glued:
        inner = glued.group("name")
        if _as_system_name(inner):
            addressed, matched, consumed = True, words[0], 1
        elif (found := _as_agent(inner, known)):
            agent, addressed, matched, consumed = found, True, words[0], 1

    # Otherwise: an optional greeting, then optionally the system name, then
    # optionally an agent name. Any prefix of that shape addresses Bolt.
    if not consumed:
        if words[0] in GREETINGS:
            consumed += 1
        if consumed < len(words) and _as_system_name(words[consumed]):
            addressed, matched = True, words[consumed]
            consumed += 1
        elif consumed and not addressed:
            consumed = 0  # a bare greeting is not an address

    # Filler may sit between the system name and the agent name: "bolt bro, scout ...".
    while consumed < len(words) and words[consumed] in FILLER and addressed:
        consumed += 1

    if consumed < len(words) and (found := _as_agent(words[consumed], known)):
        agent, addressed, matched = found, True, words[consumed]
        consumed += 1

    # ...and after it: "bolt, bro, let's work on this".
    while consumed < len(words) and words[consumed] in FILLER and addressed:
        consumed += 1

    remainder = text
    if consumed:
        # Re-slice the original text so punctuation and casing survive.
        pattern = r"^\W*" + r"\W+".join(re.escape(w) for w in words[:consumed]) + r"\W*"
        remainder = re.sub(pattern, "", text, flags=re.IGNORECASE)

    if REQUIRE_ADDRESS and not addressed:
        return Address(default_agent, text, False)
    return Address(agent, remainder.strip() or text, True, matched)

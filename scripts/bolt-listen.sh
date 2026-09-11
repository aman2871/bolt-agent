#!/bin/bash
# Always-on voice loop. Runs on the host, because a container on macOS has no
# microphone. Say "Bolt ...", "Hey Bolt ..." or an agent's name to address it;
# anything else it hears is ignored.
set -uo pipefail

BOLT_DIR="${BOLT_DIR:-$HOME/Projects/Bolt}"
LOG="${BOLT_LISTEN_LOG:-$HOME/Library/Logs/bolt-listen.log}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') bolt listen ==="

cd "$BOLT_DIR" || { echo "no such directory: $BOLT_DIR"; exit 1; }

if [ ! -x .venv/bin/bolt ]; then
    echo "no venv at $BOLT_DIR/.venv — run: python3 -m venv .venv && .venv/bin/pip install -e ."
    exit 1
fi

# KeepAlive restarts this if the mic disappears (headphones disconnecting, sleep).
exec .venv/bin/bolt listen --quiet

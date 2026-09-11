#!/bin/bash
# Bring the Bolt stack up at login.
#
# Compose's `restart: unless-stopped` only restarts containers while the Docker
# daemon is running, and on macOS the daemon is a desktop app that may not be
# up yet at login — so this waits for it, then brings the stack up itself.
set -uo pipefail

BOLT_DIR="${BOLT_DIR:-$HOME/Projects/Bolt}"
LOG="${BOLT_LOG:-$HOME/Library/Logs/bolt-autostart.log}"
DOCKER_BIN="${DOCKER_BIN:-/usr/local/bin/docker}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "=== $(date '+%Y-%m-%d %H:%M:%S') bolt autostart ==="

if [ ! -d "$BOLT_DIR" ]; then
    echo "no such directory: $BOLT_DIR"
    exit 1
fi

if ! "$DOCKER_BIN" info >/dev/null 2>&1; then
    echo "docker daemon not running, launching Docker Desktop"
    open -a Docker || { echo "could not launch Docker Desktop"; exit 1; }
fi

echo "waiting up to ${WAIT_SECONDS}s for the docker daemon"
deadline=$(( $(date +%s) + WAIT_SECONDS ))
until "$DOCKER_BIN" info >/dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "docker daemon did not come up in time"
        exit 1
    fi
    sleep 5
done
echo "docker is up: $("$DOCKER_BIN" info --format '{{.ServerVersion}}')"

cd "$BOLT_DIR" || exit 1
"$DOCKER_BIN" compose up -d
echo "stack:"
"$DOCKER_BIN" compose ps --format 'table {{.Service}}\t{{.Status}}'

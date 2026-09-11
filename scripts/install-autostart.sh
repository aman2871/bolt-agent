#!/bin/bash
# Install (or remove) the login agent that keeps the Bolt stack running.
#   scripts/install-autostart.sh            install
#   scripts/install-autostart.sh --uninstall remove
set -euo pipefail

BOLT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Other projects agents may read, colon-separated. Defaults to a sibling Chartr
# checkout when one exists, so the host CLI sees what the container sees.
if [ -z "${BOLT_EXTRA_ROOTS:-}" ] && [ -d "$HOME/Projects/Chartr" ]; then
    BOLT_EXTRA_ROOTS="$HOME/Projects/Chartr"
fi
export BOLT_EXTRA_ROOTS
AGENTS="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"

# Which agents to manage: the docker stack, and optionally the always-on mic.
WANT_LISTEN=0
UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --listen)    WANT_LISTEN=1 ;;
        --uninstall) UNINSTALL=1 ;;
        *) echo "usage: $0 [--listen] [--uninstall]"; exit 1 ;;
    esac
done

install_agent() {
    local label="$1" script="$2" plist="$3" log="$4"
    local target="$AGENTS/$label.plist"

    sed -e "s|__SCRIPT__|$BOLT_DIR/scripts/$script|g" \
        -e "s|__BOLT_DIR__|$BOLT_DIR|g" \
        -e "s|__EXTRA_ROOTS__|${BOLT_EXTRA_ROOTS:-}|g" \
        -e "s|__LOG__|$log|g" \
        "$BOLT_DIR/scripts/$plist" > "$target"
    chmod +x "$BOLT_DIR/scripts/$script"

    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true   # re-running is safe
    launchctl bootstrap "$DOMAIN" "$target"
    launchctl enable "$DOMAIN/$label"

    echo "installed $label"
    echo "  plist:  $target"
    echo "  log:    $log"
}

remove_agent() {
    local label="$1"
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    rm -f "$AGENTS/$label.plist"
    echo "removed $label"
}

STACK_LOG="$HOME/Library/Logs/bolt-autostart.log"
LISTEN_LOG="$HOME/Library/Logs/bolt-listen.log"

if [ "$UNINSTALL" = 1 ]; then
    if [ "$WANT_LISTEN" = 1 ]; then
        remove_agent com.bolt.listen          # --listen --uninstall: only the mic
    else
        remove_agent com.bolt.stack
        remove_agent com.bolt.listen
    fi
    exit 0
fi

mkdir -p "$AGENTS"
install_agent com.bolt.stack bolt-autostart.sh com.bolt.stack.plist "$STACK_LOG"

if [ "$WANT_LISTEN" = 1 ]; then
    install_agent com.bolt.listen bolt-listen.sh com.bolt.listen.plist "$LISTEN_LOG"
    echo
    echo "The first time it opens the microphone, macOS will ask for permission."
    echo "If no prompt appears, add it by hand:"
    echo "  System Settings -> Privacy & Security -> Microphone"
fi

echo
echo "  status: launchctl print $DOMAIN/com.bolt.stack | head -20"
echo "  remove: scripts/install-autostart.sh --uninstall"

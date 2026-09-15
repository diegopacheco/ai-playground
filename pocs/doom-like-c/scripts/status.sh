#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"

if is_running; then
  printf "%-10s %-6s pid %s\n" "$GAME_NAME" "UP" "$(game_pid)"
else
  printf "%-10s %-6s pid -\n" "$GAME_NAME" "DOWN"
fi

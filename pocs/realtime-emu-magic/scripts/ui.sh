#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
healthy || { printf 'Start the app first.\n' >&2; exit 1; }
if command -v open >/dev/null; then
  open "http://127.0.0.1:$APP_PORT"
elif command -v xdg-open >/dev/null; then
  xdg-open "http://127.0.0.1:$APP_PORT"
else
  printf 'Open http://127.0.0.1:%s\n' "$APP_PORT"
fi

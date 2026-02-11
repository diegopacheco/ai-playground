#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$ROOT_DIR/.run/backend.pid" ]; then
  kill "$(cat "$ROOT_DIR/.run/backend.pid")" 2>/dev/null || true
  rm -f "$ROOT_DIR/.run/backend.pid"
fi
if [ -f "$ROOT_DIR/.run/frontend.pid" ]; then
  kill "$(cat "$ROOT_DIR/.run/frontend.pid")" 2>/dev/null || true
  rm -f "$ROOT_DIR/.run/frontend.pid"
fi
pkill -f twitter_like_backend 2>/dev/null || true
pkill -f 'serve . -l 4173' 2>/dev/null || true

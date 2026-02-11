#!/usr/bin/env bash
set -euo pipefail
if [ -f .run/backend.pid ]; then
  kill "$(cat .run/backend.pid)" 2>/dev/null || true
  rm -f .run/backend.pid
fi
if [ -f .run/frontend.pid ]; then
  kill "$(cat .run/frontend.pid)" 2>/dev/null || true
  rm -f .run/frontend.pid
fi
pkill -f twitter_like_backend 2>/dev/null || true
pkill -f 'serve . -l 4173' 2>/dev/null || true

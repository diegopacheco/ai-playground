#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$ROOT/app.pid" ]; then
  kill "$(cat "$ROOT/app.pid")" 2>/dev/null || true
  rm -f "$ROOT/app.pid"
  echo "tax-service stopped"
else
  echo "no app.pid found"
fi

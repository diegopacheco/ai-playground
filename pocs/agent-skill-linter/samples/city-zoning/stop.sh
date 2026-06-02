#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
for name in server web; do
  if [ -f "$ROOT/$name.pid" ]; then
    pid="$(cat "$ROOT/$name.pid")"
    pkill -P "$pid" 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
    rm -f "$ROOT/$name.pid"
  fi
done
echo "city-zoning stopped"

#!/usr/bin/env zsh
set -euo pipefail

PORT=${PORT:-4173}
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

serve() {
  if command -v python3 >/dev/null 2>&1; then
    python3 -m http.server "$PORT" --directory "$ROOT_DIR"
  else
    python -m http.server "$PORT" --directory "$ROOT_DIR"
  fi
}

URL="http://localhost:${PORT}/index.html"

echo "Starting local server on $URL"
serve &
SERVER_PID=$!

cleanup() {
  echo "\nStopping server $SERVER_PID"
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}

trap cleanup INT TERM EXIT

sleep 1
if command -v open >/dev/null 2>&1; then
  open "$URL"
else
  echo "Open $URL in your browser"
fi

wait "$SERVER_PID"

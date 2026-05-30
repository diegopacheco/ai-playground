#!/usr/bin/env bash
# Start the Rock Paper Scissors web server
set -e

PORT="${PORT:-8000}"
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/.server.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Server already running (PID $(cat "$PIDFILE")) on http://localhost:$PORT"
  exit 0
fi

cd "$DIR"
python3 -m http.server "$PORT" >/dev/null 2>&1 &
echo $! > "$PIDFILE"
sleep 0.5

echo "Rock Paper Scissors running at http://localhost:$PORT"
echo "PID: $(cat "$PIDFILE")"
echo "Stop with: ./stop.sh"

# Try to open in default browser (macOS)
if command -v open >/dev/null 2>&1; then
  open "http://localhost:$PORT"
fi

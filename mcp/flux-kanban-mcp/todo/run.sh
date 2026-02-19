#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /tmp/grocery

cd "$DIR/server"
bun install 2>/dev/null
bun run src/index.ts &
echo $! > /tmp/grocery/server.pid

cd "$DIR/app"
bun install 2>/dev/null
bun run dev &
echo $! > /tmp/grocery/app.pid

echo "Server PID: $(cat /tmp/grocery/server.pid)"
echo "App PID:    $(cat /tmp/grocery/app.pid)"
echo "Frontend:   http://localhost:5173"
echo "AI Server:  http://localhost:3001"
echo "To stop:    ./stop.sh"
wait

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-3000}"

if [ ! -d node_modules ]; then
  bun install
fi

PORT="$PORT" bun run server.ts > server.log 2>&1 &
echo $! > server.pid

for i in $(seq 1 60); do
  if curl -s "http://localhost:${PORT}" > /dev/null; then
    echo "3D Tetris running at http://localhost:${PORT} (pid $(cat server.pid))"
    exit 0
  fi
  sleep 1
done

echo "failed to start, see server.log"
cat server.log
exit 1

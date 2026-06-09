#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set."
  echo "Run: export OPENAI_API_KEY=sk-..."
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "Installing dependencies"
  npm install
fi

echo "Starting Mind Reader on http://localhost:3434"
nohup npm run dev > dev.log 2>&1 &
echo $! > .server.pid

for i in $(seq 1 60); do
  if curl -s http://localhost:3434 > /dev/null; then
    echo "Ready at http://localhost:3434"
    echo "Play:    http://localhost:3434"
    echo "History: http://localhost:3434/history"
    exit 0
  fi
  sleep 1
done

echo "Server did not become ready in time, check dev.log"
exit 1

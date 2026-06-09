#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "Warning: OPENAI_API_KEY is not set. The UI renders, but the AI calls will return an error."
  echo "Run: export OPENAI_API_KEY=sk-..."
fi

echo "Starting Vite dev server on http://localhost:5173 ..."
npm run dev -- --host >/tmp/performativeui-poc.log 2>&1 &
echo $! > .server.pid

for i in $(seq 1 60); do
  if curl -s -o /dev/null http://localhost:5173/; then
    echo "Server is up: http://localhost:5173"
    exit 0
  fi
  sleep 1
done

echo "Server did not start in time. See /tmp/performativeui-poc.log"
exit 1

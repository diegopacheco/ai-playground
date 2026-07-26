#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs

BACKEND_PORT=8077
FRONTEND_PORT=5199

if [ ! -d backend/.venv ]; then
  echo "backend venv missing, run ./build.sh first"
  exit 1
fi
if [ ! -d frontend/node_modules ]; then
  echo "frontend deps missing, run ./build.sh first"
  exit 1
fi

if ! curl -sf http://localhost:11434/api/tags > /dev/null; then
  echo "warning: ollama is not answering on 11434, start it with: ollama serve"
fi

"$ROOT/stop.sh" > /dev/null 2>&1 || true

cd "$ROOT/backend"
../backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port $BACKEND_PORT > "$ROOT/logs/backend.log" 2>&1 &
echo $! > "$ROOT/logs/backend.pid"
cd "$ROOT"

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:$BACKEND_PORT/api/health > /dev/null; then
    break
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:$BACKEND_PORT/api/health > /dev/null; then
  echo "backend failed to start, see logs/backend.log"
  tail -20 "$ROOT/logs/backend.log"
  exit 1
fi
echo "backend up on http://127.0.0.1:$BACKEND_PORT"

cd "$ROOT/frontend"
npm run dev > "$ROOT/logs/frontend.log" 2>&1 &
echo $! > "$ROOT/logs/frontend.pid"
cd "$ROOT"

for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:$FRONTEND_PORT > /dev/null; then
    break
  fi
  sleep 1
done

if ! curl -sf http://127.0.0.1:$FRONTEND_PORT > /dev/null; then
  echo "frontend failed to start, see logs/frontend.log"
  tail -20 "$ROOT/logs/frontend.log"
  exit 1
fi

echo "frontend up on http://127.0.0.1:$FRONTEND_PORT"
echo "open http://127.0.0.1:$FRONTEND_PORT"

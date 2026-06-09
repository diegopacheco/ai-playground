#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "backend unit tests"
cd "$ROOT/backend"
go test ./...

echo "frontend typecheck"
cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install >/dev/null 2>&1
fi
npx tsc --noEmit
echo "frontend build"
npm run build >/dev/null 2>&1
echo "build ok"

PORT="${PORT:-8080}"
if curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then
  echo "health: $(curl -s "http://localhost:$PORT/api/health")"
else
  echo "backend not running; start it with ./start.sh to check /api/health"
fi

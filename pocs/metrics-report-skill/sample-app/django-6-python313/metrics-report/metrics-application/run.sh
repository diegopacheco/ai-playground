#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

PID=$(lsof -ti:3737)
if [ -n "$PID" ]; then
  kill -9 $PID
fi

cd "$SCRIPT_DIR"

if [ ! -d "node_modules" ]; then
  npm install
fi

mkdir -p public/data/history

if [ -f ../data/metrics-latest.json ]; then
  cp ../data/metrics-latest.json public/data/
fi

if [ -f ../data/history-index.json ]; then
  cp ../data/history-index.json public/data/
fi

if ls ../data/history/*.json 1>/dev/null 2>&1; then
  cp ../data/history/*.json public/data/history/
fi

npm run build

npx serve -s dist -l 3737 &

echo "Metrics report available at http://localhost:3737"

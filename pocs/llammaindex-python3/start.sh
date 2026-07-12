#!/bin/bash
set -e
cd "$(dirname "$0")"

PYTHON=python3.13
command -v $PYTHON >/dev/null 2>&1 || PYTHON=python3

if [ ! -d venv ]; then
  $PYTHON -m venv venv
fi
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
  echo "Ollama is not running on http://localhost:11434"
  echo "Start it with: ollama serve"
  exit 1
fi

ollama pull nomic-embed-text >/dev/null 2>&1 || true

nohup venv/bin/python app.py > server.log 2>&1 &
echo $! > server.pid
echo "Started app (pid $(cat server.pid)), waiting for it to become ready..."

for i in $(seq 1 60); do
  if curl -s --max-time 2 http://localhost:8000/health >/dev/null 2>&1; then
    echo "Ready at http://localhost:8000"
    exit 0
  fi
  sleep 1
done

echo "Server did not become ready in time. Check server.log"
exit 1

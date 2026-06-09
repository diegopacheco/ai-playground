#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

./stop.sh >/dev/null 2>&1 || true

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

if [ ! -f hand_landmarker.task ]; then
  curl -L -s -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

./.venv/bin/python server.py & echo $! > server.pid

for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://localhost:8000/"; then
    echo "open http://localhost:8000 in your browser"
    exit 0
  fi
  sleep 1
done

echo "server did not start"
exit 1

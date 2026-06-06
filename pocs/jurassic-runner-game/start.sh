#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

pkill -f "pose_server.py" 2>/dev/null || true
for i in $(seq 1 10); do
  if ! curl -s -o /dev/null "http://localhost:8000/"; then break; fi
  sleep 1
done

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt

if [ ! -f pose_landmarker.task ]; then
  curl -L -s -o pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
fi

./.venv/bin/python pose_server.py & echo $! > server.pid

for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://localhost:8000/"; then
    echo "open http://localhost:8000 in your browser"
    exit 0
  fi
  sleep 1
done

echo "server did not start"
exit 1

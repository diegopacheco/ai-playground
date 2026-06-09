#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f hand_landmarker.task ]; then
  curl -L -s -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

./.venv/bin/python server.py & SRV=$!
trap "kill $SRV 2>/dev/null" EXIT

UP=0
for i in $(seq 1 60); do
  if curl -s -o /dev/null "http://localhost:8000/"; then UP=1; break; fi
  sleep 1
done
if [ "$UP" != "1" ]; then echo "FAIL: http not up"; exit 1; fi

PAGE=$(curl -s "http://localhost:8000/")
case "$PAGE" in
  *"AIR FISHING"*) echo "http page ok";;
  *) echo "FAIL: page content missing"; exit 1;;
esac

./.venv/bin/python test_client.py
echo "ALL TESTS PASSED"

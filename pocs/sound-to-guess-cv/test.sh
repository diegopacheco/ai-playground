#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

BASE="http://localhost:8090"
MODEL="https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
WASM="https://cdn.jsdelivr.net/npm/@mediapipe/tasks-audio@0.10.20/wasm/audio_wasm_internal.js"

started=0
if ! curl -s -o /dev/null "$BASE/"; then
  ./start.sh >/dev/null
  started=1
fi

fail=0
check() {
  code=$(curl -s -o /dev/null -w "%{http_code}" -L "$1")
  if [ "$code" = "200" ]; then
    echo "OK    $2"
  else
    echo "FAIL  $2 (HTTP $code)"
    fail=1
  fi
}

check "$BASE/" "index.html served"
check "$BASE/app.js" "app.js served"
check "$BASE/style.css" "style.css served"
check "$MODEL" "YAMNet model reachable"
check "$WASM" "MediaPipe audio WASM reachable"

if [ "$started" = "1" ]; then
  ./stop.sh >/dev/null
fi

if [ "$fail" = "0" ]; then
  echo "ALL CHECKS PASSED"
else
  echo "SOME CHECKS FAILED"
  exit 1
fi

#!/bin/bash
cd "$(dirname "$0")"
./start.sh
body=$(curl -s http://localhost:8000/index.html)
pass=0
fail=0
check() {
  if echo "$body" | grep -q "$1"; then
    echo "PASS: $2"
    pass=$((pass+1))
  else
    echo "FAIL: $2"
    fail=$((fail+1))
  fi
}
check "<title>Statue!</title>" "page title"
check "getUserMedia" "webcam capture"
check "getImageData" "frame differencing"
check "AudioContext" "music engine"
check "STATUE!" "freeze phase"
check "DANCE!" "dance phase"
./stop.sh
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]

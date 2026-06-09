#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

./stop.sh >/dev/null 2>&1 || true

python3 -m http.server 8090 >/dev/null 2>&1 &
echo $! > server.pid

for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://localhost:8090/"; then
    echo "Sound Guesser is up"
    echo "open http://localhost:8090 in your browser and allow the microphone"
    exit 0
  fi
  sleep 1
done

echo "server did not start"
exit 1

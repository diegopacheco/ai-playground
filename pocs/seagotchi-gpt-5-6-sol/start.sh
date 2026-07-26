#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d node_modules ]; then
  npm install
fi
if [ -f .seagotchi.pid ] && [ -f .seagotchi.port ]; then
  existing_process_id="$(cat .seagotchi.pid)"
  existing_port="$(cat .seagotchi.port)"
  listener_process_id="$(lsof -nP -a -p "$existing_process_id" -iTCP:"$existing_port" -sTCP:LISTEN -t 2>/dev/null || true)"
  if [ "$listener_process_id" = "$existing_process_id" ]; then
    echo "Seagotchi is already running at http://localhost:$existing_port"
    exit 0
  fi
fi
rm -f .seagotchi.pid .seagotchi.port
port=4242
while lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; do
  port=$((port + 1))
done
./node_modules/.bin/vite --host 0.0.0.0 --port "$port" --strictPort > .seagotchi.log 2>&1 &
process_id=$!
echo "$process_id" > .seagotchi.pid
echo "$port" > .seagotchi.port
attempt=0
while [ "$attempt" -lt 60 ]; do
  if curl -fsS "http://localhost:$port" >/dev/null 2>&1; then
    echo "Seagotchi is running at http://localhost:$port"
    exit 0
  fi
  if ! kill -0 "$process_id" 2>/dev/null; then
    cat .seagotchi.log
    rm -f .seagotchi.pid .seagotchi.port
    exit 1
  fi
  attempt=$((attempt + 1))
  sleep 1
done
kill "$process_id"
rm -f .seagotchi.pid .seagotchi.port
echo "Seagotchi did not start"
exit 1

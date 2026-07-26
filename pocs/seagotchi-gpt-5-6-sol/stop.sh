#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -f .seagotchi.pid ]; then
  rm -f .seagotchi.port
  echo "Seagotchi is not running"
  exit 0
fi
process_id="$(cat .seagotchi.pid)"
port="$(cat .seagotchi.port 2>/dev/null || true)"
listener_process_id="$(lsof -nP -a -p "$process_id" -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)"
if [ "$listener_process_id" = "$process_id" ]; then
  kill "$process_id"
fi
rm -f .seagotchi.pid .seagotchi.port
echo "Seagotchi stopped"

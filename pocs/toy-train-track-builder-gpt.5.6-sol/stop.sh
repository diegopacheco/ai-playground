#!/bin/bash
set -e
cd "$(dirname "$0")"
stopped=false
for pid_file in .blockrails.pid .toy-track.pid; do
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      stopped=true
    fi
    rm "$pid_file"
  fi
done
if [ "$stopped" = true ]; then
  echo "BlockRails stopped"
else
  echo "BlockRails is not running"
fi

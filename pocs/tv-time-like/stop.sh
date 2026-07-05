#!/usr/bin/env bash
set -e
for file in .reelmark-web.pid .reelmark-server.pid; do
  if [ -f "$file" ]; then
    pid="$(<"$file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
    rm -f "$file"
  fi
done
echo "Reelmark stopped"

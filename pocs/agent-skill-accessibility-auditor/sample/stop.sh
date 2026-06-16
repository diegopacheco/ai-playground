#!/usr/bin/env bash
set -euo pipefail

PID="$(lsof -ti tcp:5188 || true)"
if [ -n "$PID" ]; then
  kill "$PID"
  echo "stopped Pixel Pantry on port 5188"
else
  echo "nothing running on port 5188"
fi

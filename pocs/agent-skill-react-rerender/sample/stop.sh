#!/usr/bin/env bash
set -euo pipefail

pid="$(lsof -ti tcp:5188 || true)"
if [ -n "$pid" ]; then
  kill "$pid"
  echo "stopped sample app on port 5188"
else
  echo "sample app not running on port 5188"
fi

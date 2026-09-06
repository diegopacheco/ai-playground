#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if pid="$(owned_pid)"; then
  kill "$pid"
  for ((i=0;i<30;i++)); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  kill -0 "$pid" 2>/dev/null && fail 'WEB did not stop within 30 seconds'
fi
rm -f "$RUN/web.pid"
printf '%s\n' 'WEB stopped'

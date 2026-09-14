#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if pid="$(owned_pid)"; then
  kill "$pid"
  for attempt in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$RUN/frontend.pid"
      printf '%s\n' 'Frontend stopped'
      exit 0
    fi
    sleep 1
  done
  fail "Frontend pid $pid did not stop"
fi
rm -f "$RUN/frontend.pid"
printf '%s\n' 'Frontend already stopped'

#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if pid="$(owned_pid)"; then
  kill "$pid"
  for ((attempt=0; attempt<30; attempt++)); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$RUN/frontend.pid"
      printf 'Frontend DOWN\n'
      exit 0
    fi
    sleep 1
  done
  fail "Frontend PID $pid did not stop within 30 seconds"
fi
rm -f "$RUN/frontend.pid"
printf 'Frontend already stopped.\n'

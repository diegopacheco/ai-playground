#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if owned; then
  pid="$(cat "$RUN/web.pid")"
  kill "$pid"
  for ((attempt=0; attempt<15; attempt++)); do
    if ! owned; then break; fi
    sleep 1
  done
  if owned; then fail 'Brisa did not stop within 15 seconds'; fi
fi
rm -f "$RUN/web.pid"
printf 'Brisa is stopped\n'

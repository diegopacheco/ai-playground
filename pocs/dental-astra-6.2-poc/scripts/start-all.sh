#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
mkdir -p .run/logs
if ! running; then
  nohup python3 -m http.server "$WEB_PORT" --bind 127.0.0.1 --directory "$ROOT" > .run/logs/web.log 2>&1 < /dev/null &
  echo "$!" > .run/web.pid
fi
for ((attempt=0;attempt<30;attempt++)); do
  if ! running; then cat .run/logs/web.log >&2; exit 1; fi
  if curl -fsS "$(service_url)/index.html" -o /dev/null 2>/dev/null; then service_url; exit 0; fi
  sleep 1
done
echo "Server did not become ready" >&2
exit 1

#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if port_up "$REPORT"; then
  echo "report server already listening on $REPORT"
  exit 0
fi
nohup python3 -m http.server "$REPORT" --bind 127.0.0.1 --directory "$ROOT/sample" >"$LOG_DIR/report.log" 2>&1 &
echo $! >"$RUN_DIR/report.pid"
wait_port "$REPORT"
echo "report server up on http://127.0.0.1:$REPORT/"

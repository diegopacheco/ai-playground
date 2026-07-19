#!/usr/bin/env bash
set -euo pipefail
if [ -f /tmp/dev-admin-console-backend.pid ]; then
  kill "$(cat /tmp/dev-admin-console-backend.pid)" 2>/dev/null || true
  rm -f /tmp/dev-admin-console-backend.pid
fi
pkill -f 'dev-admin-console.jar' 2>/dev/null || true
exit 0

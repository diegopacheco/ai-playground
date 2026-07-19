#!/usr/bin/env bash
set -euo pipefail
if [ -f /tmp/admin-console-backend.pid ]; then
  kill "$(cat /tmp/admin-console-backend.pid)" 2>/dev/null || true
  rm -f /tmp/admin-console-backend.pid
fi
pkill -f 'admin-console.jar' 2>/dev/null || true
exit 0

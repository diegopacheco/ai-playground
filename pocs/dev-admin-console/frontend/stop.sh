#!/usr/bin/env bash
set -euo pipefail
if [ -f /tmp/dev-admin-console-frontend.pid ]; then
  kill "$(cat /tmp/dev-admin-console-frontend.pid)" 2>/dev/null || true
  rm -f /tmp/dev-admin-console-frontend.pid
fi
pkill -f "astro dev" 2>/dev/null || true
exit 0

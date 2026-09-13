#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require lsof
pid="$(port_pid)"
if [ -n "$pid" ]; then
  printf 'Frontend PORT %s UP PID %s\n' "$FRONTEND" "$pid"
else
  printf 'Frontend PORT %s DOWN PID -\n' "$FRONTEND"
fi

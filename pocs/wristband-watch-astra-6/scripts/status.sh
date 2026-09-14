#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if pid="$(owned_pid)" && healthy; then
  printf 'Frontend port %s UP pid %s\n' "$FRONTEND" "$pid"
else
  printf 'Frontend port %s DOWN pid -\n' "$FRONTEND"
fi

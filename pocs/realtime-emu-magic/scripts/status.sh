#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if owned_pid && healthy; then
  printf 'APP port=%s UP pid=%s\n' "$APP_PORT" "$APP_PID"
else
  printf 'APP port=%s DOWN\n' "$APP_PORT"
fi

#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scripts/runtime.sh"
if is_running; then
  echo "Server running (PID $server_pid); log: $log_file"
else
  echo "Server is stopped"
  exit 1
fi

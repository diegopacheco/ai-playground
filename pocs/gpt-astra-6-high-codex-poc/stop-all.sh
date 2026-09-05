#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scripts/runtime.sh"
if ! is_running; then
  rm -f "$pid_file"
  echo "Server is stopped"
  exit 0
fi
kill "$server_pid"
for ((attempt = 0; attempt < 30; attempt++)); do
  if ! is_running; then
    rm -f "$pid_file"
    echo "Server stopped"
    exit 0
  fi
  sleep 1
done
echo "Server did not stop (PID $server_pid)" >&2
exit 1

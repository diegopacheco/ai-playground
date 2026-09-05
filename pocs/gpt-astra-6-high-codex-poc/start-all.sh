#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scripts/runtime.sh"
if is_running; then
  echo "Server already running (PID $server_pid)"
  exit 0
fi
bash build.sh
mkdir -p .runtime
nohup node "$project_root/server.js" "$project_root/dist" > "$log_file" 2>&1 < /dev/null &
server_pid=$!
echo "$server_pid" > "$pid_file"
for ((attempt = 0; attempt < 30; attempt++)); do
  if ! is_running; then
    cat "$log_file"
    rm -f "$pid_file"
    exit 1
  fi
  if [[ "$(cat "$log_file")" == *"Sidewalk Sessions running at"* ]] && curl --fail --silent --max-time 1 "http://127.0.0.1:${PORT:-3000}/" > /dev/null; then
    cat "$log_file"
    echo "PID $server_pid; log: $log_file"
    exit 0
  fi
  sleep 1
done
echo "Server did not become ready; log: $log_file" >&2
bash stop-all.sh
exit 1

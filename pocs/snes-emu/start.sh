#!/usr/bin/env bash
set -e
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pid_file="$project_dir/.superblue.pid"
log_file="$project_dir/.superblue.log"
port="${PORT:-9011}"
if [ -f "$pid_file" ] && kill -0 "$(sed -n '1p' "$pid_file")" 2>/dev/null; then
  echo "SuperBlue is already running at http://127.0.0.1:$port"
  exit 0
fi
nohup node "$project_dir/server.mjs" "$port" >"$log_file" 2>&1 &
server_pid=$!
echo "$server_pid" >"$pid_file"
for _ in $(seq 1 30); do
  if kill -0 "$server_pid" 2>/dev/null && grep -q "^SuperBlue running at http://127.0.0.1:$port$" "$log_file" && curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "SuperBlue started at http://127.0.0.1:$port"
    exit 0
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    sed -n '1,80p' "$log_file"
    exit 1
  fi
  sleep 1
done
echo "SuperBlue did not start within 30 seconds"
exit 1

#!/usr/bin/env bash
set -e
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
pid_file="$project_dir/.superblue.pid"
if [ ! -f "$pid_file" ]; then
  echo "SuperBlue is not running"
  exit 0
fi
server_pid=$(sed -n '1p' "$pid_file")
if kill -0 "$server_pid" 2>/dev/null; then
  kill "$server_pid"
  for _ in $(seq 1 30); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    echo "SuperBlue could not stop process $server_pid"
    exit 1
  fi
fi
rm -f "$pid_file"
echo "SuperBlue stopped"

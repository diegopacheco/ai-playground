#!/bin/bash
cd "$(dirname "$0")"

stop_pid_file() {
  local file=$1
  if [ -f "$file" ]; then
    local pid
    pid=$(cat "$file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      until ! kill -0 "$pid" 2>/dev/null; do
        sleep 1
      done
    fi
    rm -f "$file"
  fi
}

stop_port() {
  local port=$1
  local pids
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    kill $pids
  fi
}

stop_pid_file .frontend.pid
stop_pid_file .backend.pid
stop_port 8000
stop_port 8080

echo "all services stopped"

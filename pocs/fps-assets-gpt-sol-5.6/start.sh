#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

owned_server() {
  local check_pid="$1"
  local check_port="$2"
  local server_command
  server_command="$(ps -p "$check_pid" -o command= 2>/dev/null || true)"
  [[ "$server_command" == *"-m http.server $check_port"* ]]
}

if [ -f .server.pid ]; then
  remembered_pid="$(<.server.pid)"
  remembered_command="$(ps -p "$remembered_pid" -o command= 2>/dev/null || true)"
  if kill -0 "$remembered_pid" 2>/dev/null; then
    if [ -f .server.port ]; then
      remembered_port="$(<.server.port)"
      if owned_server "$remembered_pid" "$remembered_port"; then
        echo "Dustline is already running at http://localhost:$remembered_port"
        exit 0
      fi
    elif [[ "$remembered_command" =~ -m[[:space:]]+http\.server[[:space:]]+([0-9]+) ]]; then
      remembered_port="${BASH_REMATCH[1]}"
      echo "$remembered_port" >.server.port
      echo "Dustline is already running at http://localhost:$remembered_port"
      exit 0
    fi
  fi
fi

if [ -f .server.pid ]; then
  unlink .server.pid
fi
if [ -f .server.port ]; then
  unlink .server.port
fi

base_port="${DUSTLINE_PORT:-8080}"
if [[ ! "$base_port" =~ ^[0-9]+$ ]] || [ "$base_port" -lt 1 ] || [ "$base_port" -gt 65535 ]; then
  echo "DUSTLINE_PORT must be between 1 and 65535"
  exit 1
fi
port="$base_port"
max_port=65535

while [ "$port" -le "$max_port" ]; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
    port=$((port + 1))
    continue
  fi

  nohup python3 -m http.server "$port" --bind 0.0.0.0 </dev/null >.server.log 2>&1 &
  server_pid=$!
  ready=0

  for _ in {1..20}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    if curl -fsS "http://127.0.0.1:$port/index.html" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 0.1
  done

  if [ "$ready" -eq 1 ] && kill -0 "$server_pid" 2>/dev/null && owned_server "$server_pid" "$port"; then
    echo "$server_pid" >.server.pid
    echo "$port" >.server.port
    if [ "$port" -eq "$base_port" ]; then
      echo "Dustline is running at http://localhost:$port"
    else
      echo "Port $base_port is in use. Dustline is running at http://localhost:$port"
    fi
    exit 0
  fi

  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
  fi
  port=$((port + 1))
done

echo "Dustline could not find a free port from $base_port to $max_port"
exit 1

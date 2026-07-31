#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

server_pid=""
if [ -f .server.pid ]; then
  server_pid="$(<.server.pid)"
fi

server_port=""
if [ -f .server.port ]; then
  server_port="$(<.server.port)"
elif [ -n "$server_pid" ]; then
  legacy_command="$(ps -p "$server_pid" -o command= 2>/dev/null || true)"
  if [[ "$legacy_command" =~ -m[[:space:]]+http\.server[[:space:]]+([0-9]+) ]]; then
    server_port="${BASH_REMATCH[1]}"
  fi
fi

if [ -z "$server_port" ]; then
  if [ -f .server.pid ]; then
    unlink .server.pid
  fi
  echo "Dustline is not running"
  exit 0
fi

if [ -z "$server_pid" ] || ! kill -0 "$server_pid" 2>/dev/null; then
  server_pid="$(lsof -nP -iTCP:"$server_port" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true)"
fi

server_command=""
if [ -n "$server_pid" ]; then
  server_command="$(ps -p "$server_pid" -o command= 2>/dev/null || true)"
fi

if [ -n "$server_pid" ] && [[ "$server_command" == *"-m http.server $server_port"* ]]; then
  kill "$server_pid"
  for _ in {1..20}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL "$server_pid"
  fi
fi

if [ -f .server.pid ]; then
  unlink .server.pid
fi
if [ -f .server.port ]; then
  unlink .server.port
fi

if [ -n "$server_pid" ] && [[ "$server_command" == *"-m http.server $server_port"* ]]; then
  echo "Dustline stopped on port $server_port"
else
  echo "Dustline was not running on remembered port $server_port"
fi

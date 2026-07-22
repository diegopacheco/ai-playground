#!/usr/bin/env bash

if [ ! -f .server.pid ]; then
  printf 'Angry Bode is not running\n'
  exit 0
fi
server_pid="$(cat .server.pid)"
if kill -0 "$server_pid" 2>/dev/null; then
  kill "$server_pid"
fi
rm -f .server.pid .server.port
printf 'Angry Bode stopped\n'

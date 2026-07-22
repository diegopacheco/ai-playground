#!/usr/bin/env bash

game_port="${PORT:-5188}"
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  printf 'Angry Bode is already running at http://localhost:%s\n' "$(cat .server.port)"
  exit 0
fi
if [ ! -d node_modules/three ]; then
  npm install
fi
nohup python3 -m http.server "$game_port" --bind 127.0.0.1 > .server.log 2>&1 </dev/null &
printf '%s' "$!" > .server.pid
printf '%s' "$game_port" > .server.port
sleep 1
check_count=0
while ! curl -fsS "http://127.0.0.1:$game_port/index.html" >/dev/null 2>&1; do
  if ! kill -0 "$(cat .server.pid)" 2>/dev/null; then
    printf 'Angry Bode could not start\n'
    cat .server.log
    rm -f .server.pid .server.port
    exit 1
  fi
  check_count=$((check_count + 1))
  if [ "$check_count" -ge 10 ]; then
    printf 'Angry Bode could not start\n'
    cat .server.log
    rm -f .server.pid .server.port
    exit 1
  fi
  sleep 1
done
printf 'Angry Bode is running at http://localhost:%s\n' "$game_port"

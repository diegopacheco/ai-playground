#!/usr/bin/env bash
if [ -f /tmp/ybyra.pid ] && kill -0 "$(< /tmp/ybyra.pid)" 2>/dev/null; then
  echo "Ybyrá is already running at http://127.0.0.1:8090"
  exit 0
fi
nohup python3 -m http.server 8090 --bind 127.0.0.1 > /tmp/ybyra.log 2>&1 &
echo $! > /tmp/ybyra.pid
for attempt in {1..10}; do
  if curl -sS http://127.0.0.1:8090/index.html 2>/dev/null | rg -q "<title>Ybyrá"; then
    echo "Ybyrá running at http://127.0.0.1:8090"
    exit 0
  fi
  sleep 1
done
echo "Ybyrá could not start. Read /tmp/ybyra.log"
exit 1

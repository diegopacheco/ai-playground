#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./stop.sh > /dev/null 2>&1 || true
for attempt in $(seq 1 15); do
  if ! lsof -ti tcp:8099 > /dev/null 2>&1; then
    break
  fi
  sleep 1
done
if lsof -ti tcp:8099 > /dev/null 2>&1; then
  echo "port 8099 is still held by another process, refusing to start a second backend"
  lsof -i tcp:8099
  exit 1
fi
mvn -q -B -DskipTests package
nohup java -jar target/dev-admin-console.jar > /tmp/dev-admin-console-backend.log 2>&1 &
echo $! > /tmp/dev-admin-console-backend.pid
for attempt in $(seq 1 90); do
  if curl -fsS http://localhost:8099/actuator/health >/dev/null 2>&1; then
    echo "backend ready on http://localhost:8099"
    exit 0
  fi
  sleep 1
done
tail -30 /tmp/dev-admin-console-backend.log
exit 1

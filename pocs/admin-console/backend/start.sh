#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mvn -q -B -DskipTests package
nohup java -jar target/admin-console.jar > /tmp/admin-console-backend.log 2>&1 &
echo $! > /tmp/admin-console-backend.pid
for attempt in $(seq 1 90); do
  if curl -fsS http://localhost:8099/actuator/health >/dev/null 2>&1; then
    echo "backend ready on http://localhost:8099"
    exit 0
  fi
  sleep 1
done
tail -30 /tmp/admin-console-backend.log
exit 1

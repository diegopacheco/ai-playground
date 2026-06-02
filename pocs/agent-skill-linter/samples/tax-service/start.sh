#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mvn -q -DskipTests package
java -jar target/tax-service-1.0.0.jar > "$ROOT/app.log" 2>&1 &
echo $! > "$ROOT/app.pid"
for i in $(seq 1 60); do
  if curl -sf http://localhost:8080/api/tax/health > /dev/null; then
    echo "tax-service up on http://localhost:8080"
    exit 0
  fi
  sleep 1
done
echo "tax-service failed to start, see app.log"
exit 1

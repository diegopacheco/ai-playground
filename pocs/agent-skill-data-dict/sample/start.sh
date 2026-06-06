#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mvn -q clean package -DskipTests
nohup java -jar target/petshop-1.0.0.jar > app.log 2>&1 &
echo $! > app.pid
for i in $(seq 1 60); do
  if curl -sf http://localhost:8080/api/tables > /dev/null 2>&1; then
    echo "petshop started on http://localhost:8080 (pid $(cat app.pid))"
    exit 0
  fi
  sleep 1
done
echo "petshop did not start in time; see app.log"
exit 1

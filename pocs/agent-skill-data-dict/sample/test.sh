#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mvn -q clean package -DskipTests
nohup java -jar target/petshop-1.0.0.jar > app.log 2>&1 &
pid=$!
echo "$pid" > app.pid

ready=0
for i in $(seq 1 60); do
  if curl -sf http://localhost:8080/api/tables > /dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -ne 1 ]; then
  echo "FAIL: app did not start"
  kill "$pid" 2>/dev/null || true
  rm -f app.pid
  exit 1
fi

tables=$(curl -sf http://localhost:8080/api/tables)
echo "tables in running database: $tables"

kill "$pid" 2>/dev/null || true
rm -f app.pid

ok=1
for t in OWNER PET VISIT SUPPLIER PRODUCT APPOINTMENT VACCINATION; do
  if ! echo "$tables" | grep -q "\"$t\""; then
    echo "MISSING table: $t"
    ok=0
  fi
done

if [ "$ok" -eq 1 ]; then
  echo "PASS: all 7 tables created across Liquibase, JDBC and Hibernate"
  exit 0
fi
echo "FAIL: some tables were not created"
exit 1

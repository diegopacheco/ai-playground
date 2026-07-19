#!/usr/bin/env bash
set -euo pipefail
container="${1:-admin-console-demo-cassandra}"
rows="${2:-300}"
file=/tmp/cassandra-rows.cql
: > "$file"
echo "USE shop;" >> "$file"
for index in $(seq 1 "$rows"); do
  customer=$(( index % 40 ))
  echo "INSERT INTO events_by_customer (customer_id, event_time, event_id, event_type, payload) VALUES ($customer, '2026-01-01T00:00:00Z'+${index}s, uuid(), 'view', 'payload-$index');" >> "$file"
done
for index in $(seq 1 100); do
  customer=$(( index % 40 ))
  echo "INSERT INTO sessions (session_id, customer_id, started_at, user_agent) VALUES (uuid(), $customer, toTimestamp(now()), 'agent-$index');" >> "$file"
done
podman cp "$file" "$container":/tmp/cassandra-rows.cql
podman exec "$container" cqlsh -f /tmp/cassandra-rows.cql

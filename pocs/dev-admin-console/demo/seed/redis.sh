#!/usr/bin/env bash
set -euo pipefail
container="${1:-dev-admin-console-demo-redis}"
run() { podman exec "$container" redis-cli "$@" > /dev/null; }
run FLUSHALL
run SET config:app:name "dev-admin-console-demo"
run SET config:app:version "1.0.0"
run SET counter:visits 4821
for index in $(seq 1 60); do
  run SET "cache:customer:$index" "{\"id\":$index,\"name\":\"Customer $index\"}"
done
for index in $(seq 1 40); do
  run HSET customer:emails "$index" "customer$index@example.com"
done
run HSET session:abc123 user "diego" ip "10.0.0.7" agent "firefox" expires "3600"
run HSET session:def456 user "reader" ip "10.0.0.9" agent "chrome" expires "1800"
run RPUSH queue:emails "welcome:1" "welcome:2" "receipt:3" "receipt:4" "digest:5"
run SADD tags:active "brazil" "europe" "premium" "beta"
run ZADD leaderboard 120 "diego" 340 "ana" 275 "bruno" 95 "carla"
run XADD events:orders '*' type placed order 1001
run XADD events:orders '*' type paid order 1001
run XADD events:orders '*' type shipped order 1001
podman exec "$container" redis-cli DBSIZE

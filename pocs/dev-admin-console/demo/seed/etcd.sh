#!/usr/bin/env bash
set -euo pipefail
container="${1:-admin-console-demo-etcd}"
put() { podman exec "$container" etcdctl put "$1" "$2" > /dev/null; }
podman exec "$container" etcdctl del "" --from-key=true > /dev/null
put /config/app/name "admin-console"
put /config/app/version "1.0.0"
put /config/app/log-level "info"
put /config/database/host "postgres.internal"
put /config/database/port "5432"
put /config/database/pool-size "8"
put /config/features/dark-mode "false"
put /config/features/audit "true"
put /service/api/instance-1 "10.0.0.11:8080"
put /service/api/instance-2 "10.0.0.12:8080"
put /service/worker/instance-1 "10.0.0.21:9000"
for index in $(seq 1 40); do
  put "/leases/tenant-$index/token" "token-value-$index"
done
podman exec "$container" etcdctl get "" --from-key=true --keys-only | grep -c . || true

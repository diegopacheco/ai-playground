#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! podman info > /dev/null 2>&1; then
  exit 0
fi
podman-compose down -v > /dev/null 2>&1 || true
for container in mysql postgres cassandra redis etcd kafka elasticsearch; do
  podman rm -f "dev-admin-console-demo-$container" > /dev/null 2>&1 || true
done
podman network rm demo_default > /dev/null 2>&1 || true
echo "demo environment stopped"

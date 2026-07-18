#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! podman info > /dev/null 2>&1; then
  podman machine start
  for attempt in $(seq 1 60); do
    podman info > /dev/null 2>&1 && break
    sleep 1
  done
fi

podman-compose up -d > /dev/null
for attempt in $(seq 1 60); do
  podman exec admin-console-metadata pg_isready -U admin_console > /dev/null 2>&1 && break
  sleep 1
done

./backend/start.sh
./demo/demo-start.sh

echo ""
echo "running integration tests"
cd backend
mvn -B -Dexcluded.groups= -Dincluded.groups=integration-test test

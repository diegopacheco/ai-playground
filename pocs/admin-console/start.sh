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
podman info > /dev/null

podman-compose up -d --force-recreate > /dev/null
for attempt in $(seq 1 60); do
  if podman exec admin-console-metadata pg_isready -U admin_console > /dev/null 2>&1; then
    echo "metadata postgres ready on 5433"
    break
  fi
  sleep 1
done

./backend/start.sh
./frontend/start.sh
./links.sh

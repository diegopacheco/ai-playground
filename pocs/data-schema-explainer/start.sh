#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
podman-compose up -d --build --force-recreate
printf "waiting for website on http://localhost:8080\n"
for i in $(seq 1 90); do
  if curl -sf http://localhost:8080/api/samples >/dev/null 2>&1; then
    printf "website is up at http://localhost:8080\n"
    exit 0
  fi
  sleep 1
done
printf "website did not become ready in time\n"
exit 1

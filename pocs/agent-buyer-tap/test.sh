#!/usr/bin/env bash
set -e
node shared/selftest.mjs
mkdir -p keys printscreens
podman-compose up -d --build key-directory merchant
for i in $(seq 1 120); do
  if curl -sf http://localhost:8802/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
podman-compose build buyer-agent
out=$(podman-compose run --rm buyer-agent 2>&1)
echo "$out"
echo "$out" | grep -q "PURCHASE ACCEPTED"
echo "E2E OK"

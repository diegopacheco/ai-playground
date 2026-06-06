#!/usr/bin/env bash
set -e
mkdir -p keys printscreens
podman-compose up -d --build key-directory merchant
for i in $(seq 1 120); do
  if curl -sf http://localhost:8801/health >/dev/null 2>&1 && curl -sf http://localhost:8802/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -sf http://localhost:8801/health >/dev/null
curl -sf http://localhost:8802/health >/dev/null
echo "key-directory up on http://localhost:8801/.well-known/jwks.json"
echo "merchant + storefront up on http://localhost:8802"

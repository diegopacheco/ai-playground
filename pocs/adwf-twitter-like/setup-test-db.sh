#!/bin/bash

podman rm -f twitter_test_db 2>/dev/null || true

podman run -d \
  --name twitter_test_db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=twitter_test \
  -p 5433:5432 \
  postgres:16-alpine

max_wait=30
elapsed=0
while ! podman exec twitter_test_db pg_isready -U postgres > /dev/null 2>&1; do
  if [ $elapsed -ge $max_wait ]; then
    echo "Database failed to start within ${max_wait} seconds"
    exit 1
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done

echo "Test database is ready on port 5433"

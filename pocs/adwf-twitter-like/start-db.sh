#!/bin/bash
podman run -d \
  --name twitter_postgres \
  -e POSTGRES_DB=twitter_db \
  -e POSTGRES_USER=twitter_user \
  -e POSTGRES_PASSWORD=twitter_pass \
  -p 5432:5432 \
  docker.io/library/postgres:18

echo "Waiting for PostgreSQL to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
  if podman exec twitter_postgres pg_isready -U twitter_user -d twitter_db > /dev/null 2>&1; then
    echo "PostgreSQL is ready!"
    exit 0
  fi
  ATTEMPT=$((ATTEMPT + 1))
  sleep 1
done

echo "PostgreSQL failed to start within expected time"
exit 1

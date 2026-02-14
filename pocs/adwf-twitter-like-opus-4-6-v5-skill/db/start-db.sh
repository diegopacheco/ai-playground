#!/bin/bash
podman stop twitter-postgres 2>/dev/null
podman rm twitter-postgres 2>/dev/null

podman run -d \
  --name twitter-postgres \
  -e POSTGRES_USER=twitter \
  -e POSTGRES_PASSWORD=twitter123 \
  -e POSTGRES_DB=twitter \
  -p 5432:5432 \
  docker.io/library/postgres:18

while ! podman exec twitter-postgres pg_isready -U twitter -d twitter > /dev/null 2>&1; do
  sleep 1
done

echo "PostgreSQL is ready."

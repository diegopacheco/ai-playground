#!/bin/bash
podman run -d \
  --name blogdb-postgres \
  -e POSTGRES_DB=blogdb \
  -e POSTGRES_USER=bloguser \
  -e POSTGRES_PASSWORD=blogpass \
  -p 5432:5432 \
  docker.io/library/postgres:17

echo "Waiting for PostgreSQL to be ready..."
until podman exec blogdb-postgres pg_isready -U bloguser -d blogdb > /dev/null 2>&1; do
  sleep 1
done
echo "PostgreSQL is ready."

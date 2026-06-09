#!/bin/bash
set -e

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-..."
  exit 1
fi

podman-compose up -d --build --force-recreate

echo "Waiting for PostgreSQL to be ready..."
until podman exec ai-postgres-db pg_isready -U postgres -d aidb >/dev/null 2>&1; do
  sleep 1
done

until podman exec ai-postgres-db psql -U postgres -d aidb -tAc "SELECT 1 FROM reviews LIMIT 1" >/dev/null 2>&1; do
  sleep 1
done

echo "PostgreSQL 19 is ready with the llm_classify() SQL function loaded."

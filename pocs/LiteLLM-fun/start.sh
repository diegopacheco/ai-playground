#!/bin/bash
podman-compose up -d
while ! podman exec $(podman ps -q -f name=postgres) pg_isready -U litellm > /dev/null 2>&1; do
  sleep 1
done
echo "PostgreSQL is ready"
source .venv/bin/activate
export DATABASE_URL="postgresql://litellm:litellm@localhost:5432/litellm"
export LITELLM_MASTER_KEY="sk-1234"
litellm --config litellm_config.yaml --port 4000

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
app_port="${APP_PORT:-8082}"
export COMPOSE_PROJECT_NAME="java25-aws-floci-s3-vectors"
export APP_PORT="$app_port"
export FLOCI_PORT="${FLOCI_PORT:-4568}"
./stop.sh >/dev/null
if ! podman-compose up --build -d; then
  ./stop.sh >/dev/null
  exit 1
fi
for attempt in $(seq 1 60); do
  if curl -fsS "http://localhost:$app_port/api/v1/status" >/dev/null 2>&1; then
    echo "Application: http://localhost:$app_port"
    echo "Swagger UI: http://localhost:$app_port/swagger-ui.html"
    exit 0
  fi
  sleep 1
done
podman-compose logs
./stop.sh >/dev/null
exit 1

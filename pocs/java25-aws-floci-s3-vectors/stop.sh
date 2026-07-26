#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project_name="java25-aws-floci-s3-vectors"
export COMPOSE_PROJECT_NAME="$project_name"
containers="$(podman ps -aq --filter "label=io.podman.compose.project=$project_name")"
if [ -z "$containers" ]; then
  echo "Application is not running"
  exit 0
fi
podman-compose down >/dev/null 2>&1 || true
containers="$(podman ps -aq --filter "label=io.podman.compose.project=$project_name")"
if [ -n "$containers" ]; then
  echo "Could not stop application"
  exit 1
fi
echo "Application stopped"

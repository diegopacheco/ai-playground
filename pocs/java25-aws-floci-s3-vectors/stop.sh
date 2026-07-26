#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project_name="java25-aws-floci-s3-vectors"
ui_container="java25-aws-floci-s3-vectors-ui"
export COMPOSE_PROJECT_NAME="$project_name"
stopped="false"
if podman container exists "$ui_container"; then
  podman rm -f "$ui_container" >/dev/null
  stopped="true"
fi
containers="$(podman ps -aq --filter "label=io.podman.compose.project=$project_name")"
if [ -n "$containers" ]; then
  podman-compose down >/dev/null 2>&1 || true
  stopped="true"
fi
containers="$(podman ps -aq --filter "label=io.podman.compose.project=$project_name")"
if [ -n "$containers" ]; then
  echo "Could not stop application"
  exit 1
fi
if [ "$stopped" = "true" ]; then
  echo "Application and Floci UI stopped"
else
  echo "Application is not running"
fi

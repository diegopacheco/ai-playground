#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ui_container="java25-aws-floci-s3-vectors-ui"
floci_container="java25-aws-floci-s3-vectors_floci_1"
network="java25-aws-floci-s3-vectors_default"
ui_port="${FLOCI_UI_PORT:-4500}"
if ! podman container exists "$floci_container"; then
  echo "Run ./start.sh before starting Floci UI"
  exit 1
fi
if [ "$(podman inspect "$floci_container" --format '{{.State.Running}}')" != "true" ]; then
  echo "Run ./start.sh before starting Floci UI"
  exit 1
fi
if podman container exists "$ui_container"; then
  podman rm -f "$ui_container" >/dev/null
fi
podman pull docker.io/floci/floci-ui:latest
if ! podman run -d \
  --name "$ui_container" \
  --network "$network" \
  -p "$ui_port:4500" \
  -e FLOCI_ENDPOINT=http://floci:4566 \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=test \
  -e AWS_SECRET_ACCESS_KEY=test \
  docker.io/floci/floci-ui:latest >/dev/null; then
  podman rm -f "$ui_container" >/dev/null 2>&1 || true
  exit 1
fi
for attempt in $(seq 1 60); do
  if curl -fsS "http://localhost:$ui_port" >/dev/null 2>&1; then
    echo "Floci UI: http://localhost:$ui_port"
    exit 0
  fi
  sleep 1
done
podman logs "$ui_container"
podman rm -f "$ui_container" >/dev/null
exit 1

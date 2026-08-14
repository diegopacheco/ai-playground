#!/usr/bin/env bash

set -euo pipefail

image="docker.io/kestra/kestra:latest"
container="kestra"

echo "Kestra is starting. Open http://localhost:8080/ui/ after startup completes."

socket="$(podman info --format '{{.Host.RemoteSocket.Path}}')"
socket="${socket#unix://}"

if podman container exists "$container"; then
  echo "Container $container already exists. Remove it before running this script."
  exit 1
fi

cleanup() {
  trap - INT TERM
  podman rm --force "$container" >/dev/null 2>&1 || true
  podman image rm --force "$image" >/dev/null 2>&1 || true
  exit 130
}

trap cleanup INT TERM

set +e
podman run --pull=always --interactive --tty --rm \
  --publish 8080:8080 \
  --user root \
  --name "$container" \
  --volume kestra_data:/app/storage \
  --volume "$socket:/var/run/docker.sock" \
  "$image" server local
status="$?"
set -e

trap - INT TERM

if [ "$status" -eq 130 ]; then
  cleanup
fi

exit "$status"

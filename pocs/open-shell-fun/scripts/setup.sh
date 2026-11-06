#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require bun
require podman

if ! command -v openshell >/dev/null 2>&1; then
  log "installing OpenShell CLI and local gateway"
  curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
fi
require openshell

if ! podman image exists "$SANDBOX_IMAGE"; then
  log "building sandbox image $SANDBOX_IMAGE"
  podman build -q -t "$SANDBOX_IMAGE" "$ROOT/sandbox" >/dev/null
fi

log "installing backend dependencies"
(cd "$ROOT/backend" && bun install --silent)
log "installing frontend dependencies"
(cd "$ROOT/frontend" && bun install --silent)

log "setup done"

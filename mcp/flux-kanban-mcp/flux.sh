#!/usr/bin/env bash
set -euo pipefail

IMAGE="sirsjg/flux-mcp:latest"

if ! command -v podman >/dev/null 2>&1; then
  echo "Docker is required. Install Docker Desktop: https://www.podman.com/get-started" >&2
  exit 1
fi

echo "Pulling Flux image..."
podman pull "$IMAGE"

echo "Starting Flux web/API..."
if podman ps -a --format '{{.Names}}' | grep -q '^flux-web$'; then
  podman rm -f flux-web >/dev/null
fi
podman run -d --userns=keep-id -p 3000:3000 -v flux-data:/app/packages/data -v flux-blobs:/home/flux -e FLUX_DATA=/app/packages/data/flux.sqlite --name flux-web "$IMAGE" bun packages/server/dist/index.js

echo ""
echo "Flux web UI is running: http://localhost:3000"
echo ""
echo "Starting MCP server (Claude/Codex)..."
echo "Press Ctrl+C to stop the MCP server"
echo ""
podman run -i --userns=keep-id --rm -v flux-data:/app/packages/data -v flux-blobs:/home/flux -e FLUX_DATA=/app/packages/data/flux.sqlite "$IMAGE" bun packages/mcp/dist/index.js

#!/usr/bin/env bash
set -euo pipefail

IMAGE="sirsjg/flux-mcp:latest"

if ! command -v podman >/dev/null 2>&1; then
  echo "podman is required. Install podman: https://podman.io/get-started" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "claude cli is required. Install Claude Code: https://claude.ai/code" >&2
  exit 1
fi

podman pull "$IMAGE"

if podman ps -a --format '{{.Names}}' | grep -q '^flux-web$'; then
  podman rm -f flux-web >/dev/null
fi

podman run -d --userns=keep-id -p 3000:3000 \
  -v flux-data:/app/packages/data \
  -v flux-blobs:/home/flux \
  -e FLUX_DATA=/app/packages/data/flux.sqlite \
  --name flux-web "$IMAGE" bun packages/server/dist/index.js

claude mcp add flux -- podman run -i --userns=keep-id --rm \
  -v flux-data:/app/packages/data \
  -v flux-blobs:/home/flux \
  -e FLUX_DATA=/app/packages/data/flux.sqlite \
  "$IMAGE" bun packages/mcp/dist/index.js

echo "Flux web UI: http://localhost:3000"
echo "MCP server registered as 'flux' in Claude Code"

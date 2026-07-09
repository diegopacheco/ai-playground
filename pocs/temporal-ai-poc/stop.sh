#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -f worker.pid ]; then
  kill "$(cat worker.pid)" 2>/dev/null || true
  rm -f worker.pid
  echo "Worker stopped"
fi

podman-compose down
echo "Temporal server stopped"

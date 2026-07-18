#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

./frontend/stop.sh
./backend/stop.sh

if podman info > /dev/null 2>&1; then
  podman rm -f admin-console-metadata > /dev/null 2>&1 || true
fi
echo "admin console stopped"

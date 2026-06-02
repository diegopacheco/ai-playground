#!/usr/bin/env bash
set -euo pipefail
SITE="$(cd "$(dirname "$0")" && pwd)"
cd "$SITE"
podman-compose down
echo "lint site stopped"

#!/usr/bin/env bash
set -euo pipefail
SITE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$PWD}"
TARGET="$(cd "$TARGET" && pwd)"
if [ ! -f "$TARGET/.lint/report.json" ]; then
  echo "no .lint/report.json in $TARGET; run /lint first"
  exit 1
fi
export LINT_TARGET="$TARGET"
(cd "$SITE/backend" && mvn -q -DskipTests package)
(cd "$SITE/frontend" && bun install && bun run build)
cd "$SITE"
podman-compose up -d --build
for i in $(seq 1 90); do
  if curl -sf http://localhost:8089/api/health > /dev/null; then break; fi
  sleep 1
done
for i in $(seq 1 60); do
  if curl -sf http://localhost:8088 > /dev/null; then
    echo "lint report at http://localhost:8088"
    exit 0
  fi
  sleep 1
done
echo "frontend failed to start"
exit 1

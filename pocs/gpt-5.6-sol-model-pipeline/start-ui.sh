#!/usr/bin/env bash
set -euo pipefail
port="${UI_PORT:-8091}"
if [[ ! -f model/artifacts/iris-network.pt ]]; then
  echo "Model artifact not found. Run ./train.sh first."
  exit 1
fi
podman-compose up -d --build --force-recreate ui
for attempt in {1..60}; do
  if curl -fsS "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo "Inference UI is ready at http://localhost:${port}"
    exit 0
  fi
  sleep 1
done
echo "Inference UI did not become ready"
exit 1

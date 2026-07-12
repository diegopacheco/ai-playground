#!/usr/bin/env bash
set -euo pipefail
podman-compose up -d temporal
podman-compose up -d --build --force-recreate worker
for attempt in {1..60}; do
  if curl -fsS http://localhost:8233 >/dev/null 2>&1; then
    break
  fi
  if [[ "$attempt" == "60" ]]; then
    echo "Temporal did not become ready"
    exit 1
  fi
  sleep 1
done
podman-compose run --rm worker python -m model_pipeline.trigger
echo "Temporal UI is ready at http://localhost:8233"
if [[ "${OPEN_BROWSER:-1}" == "1" ]] && command -v open >/dev/null 2>&1; then
  open http://localhost:8233
fi

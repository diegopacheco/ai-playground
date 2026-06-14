#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
podman-compose up -d --build --force-recreate
echo "Prometheus UI: http://localhost:9090"
echo "Alerts:        http://localhost:9090/alerts"
echo "Sample app:    http://localhost:8000/metrics"

#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

podman-compose up -d --build

attempts=0
until curl -sf http://localhost:8080/api/repos >/dev/null 2>&1; do
  attempts=$((attempts + 1))
  if [ "$attempts" -ge 300 ]; then
    echo "backend did not become ready"
    exit 1
  fi
  sleep 1
done
echo "backend ready at http://localhost:8080"

attempts=0
until curl -sf http://localhost:5173 >/dev/null 2>&1; do
  attempts=$((attempts + 1))
  if [ "$attempts" -ge 180 ]; then
    echo "frontend did not become ready"
    exit 1
  fi
  sleep 1
done
echo "ui ready at http://localhost:5173"

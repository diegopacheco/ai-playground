#!/usr/bin/env bash
set -euo pipefail
PIDS="$(lsof -ti tcp:8082 -sTCP:LISTEN 2>/dev/null || true)"
for PID in $PIDS; do
  COMMAND="$(ps -p "$PID" -o command= 2>/dev/null || true)"
  case "$COMMAND" in
    *temporal-java-25-poc-1.0.0.jar*) kill "$PID" ;;
  esac
done
podman-compose -f compose.yml down >/dev/null 2>&1 || true

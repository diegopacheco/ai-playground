#!/usr/bin/env bash
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -f /tmp/qa2pw-web.pid ]; then
  PID=$(cat /tmp/qa2pw-web.pid)
  if kill -0 "$PID" 2>/dev/null; then
    echo "stopping web (pid $PID)"
    kill "$PID" 2>/dev/null || true
    for i in $(seq 1 5); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f /tmp/qa2pw-web.pid
fi

if [ -f /tmp/qa2pw-ollama.pid ]; then
  PID=$(cat /tmp/qa2pw-ollama.pid)
  if kill -0 "$PID" 2>/dev/null; then
    echo "stopping ollama (pid $PID) — only stopping the one ./run.sh started"
    kill "$PID" 2>/dev/null || true
    for i in $(seq 1 5); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 1
    done
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f /tmp/qa2pw-ollama.pid
else
  echo "ollama: not started by run.sh; leaving it alone"
fi

if command -v podman-compose >/dev/null 2>&1 && [ -f infra/podman-compose.yml ]; then
  echo "tearing down podman-compose stack"
  podman-compose -f infra/podman-compose.yml down 2>/dev/null || true
fi

echo "qa2pw stopped"

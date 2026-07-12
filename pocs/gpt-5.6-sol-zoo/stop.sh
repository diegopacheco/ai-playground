#!/usr/bin/env bash
set -e
if [ -f .server.pid ]; then
  kill "$(cat .server.pid)" 2>/dev/null || true
  rm -f .server.pid
fi
echo "Terra Vivo stopped"

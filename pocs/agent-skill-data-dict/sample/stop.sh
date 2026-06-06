#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f app.pid ]; then
  kill "$(cat app.pid)" 2>/dev/null || true
  rm -f app.pid
  echo "petshop stopped"
else
  echo "no app.pid found"
fi

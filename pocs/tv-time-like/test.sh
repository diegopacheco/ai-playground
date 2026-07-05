#!/usr/bin/env bash
set -e
bun test
bun run check
bun run build
./start.sh
trap './stop.sh' EXIT
curl -sf http://127.0.0.1:3001/api/health
curl -sf 'http://127.0.0.1:3001/api/search?q=Severance' >/dev/null
curl -sf http://127.0.0.1:3001/api/library >/dev/null
echo
echo "All checks passed"

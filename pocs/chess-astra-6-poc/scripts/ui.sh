#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[ -n "$(port_pid)" ] || fail "Frontend is stopped; run ./scripts/start-all.sh"
url="http://127.0.0.1:$FRONTEND"
if command -v open >/dev/null 2>&1; then open "$url"
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$url"
else fail "Open $url in your browser"
fi

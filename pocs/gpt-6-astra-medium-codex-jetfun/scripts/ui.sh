#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
[ -n "$(port_pid)" ] || fail 'Run ./scripts/start-all.sh first'
if command -v open >/dev/null; then open "http://localhost:$WEB"; else xdg-open "http://localhost:$WEB"; fi

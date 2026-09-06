#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
healthy || fail 'Brisa is not running. Run ./scripts/start-all.sh first.'
if command -v open >/dev/null; then open "http://localhost:$WEB"; elif command -v xdg-open >/dev/null; then xdg-open "http://localhost:$WEB"; else fail "Open http://localhost:$WEB in your browser"; fi

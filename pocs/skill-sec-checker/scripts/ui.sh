#!/usr/bin/env bash
set -euo pipefail
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if port_up "$REPORT"; then
  open "http://127.0.0.1:$REPORT/" 2>/dev/null || xdg-open "http://127.0.0.1:$REPORT/"
else
  open "$ROOT/sample/index.html" 2>/dev/null || xdg-open "$ROOT/sample/index.html"
fi

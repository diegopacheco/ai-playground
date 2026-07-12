#!/usr/bin/env bash
set -euo pipefail
port="${UI_PORT:-8091}"
./start-ui.sh
if [[ "${OPEN_BROWSER:-1}" == "1" ]] && command -v open >/dev/null 2>&1; then
  open "http://localhost:${port}"
fi

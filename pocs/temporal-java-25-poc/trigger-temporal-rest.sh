#!/usr/bin/env bash
set -euo pipefail
SYMBOL="${1:-AAPL}"
COMPANY="${2:-Apple}"
curl -s -X POST "http://localhost:8082/api/research/trigger" -H "Content-Type: application/json" -d "{\"symbol\":\"$SYMBOL\",\"company\":\"$COMPANY\"}"
printf '\n'

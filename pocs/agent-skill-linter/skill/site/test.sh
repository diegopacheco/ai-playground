#!/usr/bin/env bash
set -euo pipefail
curl -sf http://localhost:8089/api/health > /dev/null && echo "backend ok"
curl -sf http://localhost:8088 > /dev/null && echo "frontend ok"

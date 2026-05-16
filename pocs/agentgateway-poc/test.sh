#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

curl -sf "http://localhost:8080/v1/chat/completions" \
  -H "content-type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "say hello in 5 words"}]
  }' | tee /tmp/agentgateway-test.json

echo
echo "---"
./app-cli.sh "what is 2 + 2?"

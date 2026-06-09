#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

BASE="http://localhost:5173"

echo "1. Checking the page loads..."
CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/")
if [ "$CODE" != "200" ]; then
  echo "FAIL: page returned HTTP $CODE"
  exit 1
fi
echo "   OK: HTTP 200"

echo "2. Calling the OpenAI proxy /api/chat..."
RESP=$(curl -s -X POST "$BASE/api/chat" -H 'content-type: application/json' \
  -d '{"prompt":"Say hello in five words.","model":"gpt-4o-mini"}')
echo "   Response: $RESP"

if echo "$RESP" | grep -q '"text"'; then
  echo "   OK: got an AI completion."
elif echo "$RESP" | grep -q 'OPENAI_API_KEY'; then
  echo "   OK: proxy is wired (set OPENAI_API_KEY to get a real completion)."
else
  echo "FAIL: unexpected proxy response."
  exit 1
fi

echo "All checks passed."

#!/bin/bash
set -e
cd "$(dirname "$0")"

BASE=http://localhost:8000

echo "1. Health check"
curl -s $BASE/health
echo

echo
echo "2. Query: How does the on-call rotation work?"
curl -s -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does the on-call rotation work?"}' \
  | python3 -m json.tool

echo
echo "3. Query with empty question (should 400)"
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question":""}'

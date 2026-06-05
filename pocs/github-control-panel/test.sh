#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
BASE=http://localhost:8080/api

echo "adding repo octocat/Hello-World"
curl -sf -X POST "$BASE/repos" -H "Content-Type: application/json" -d '{"repos":["octocat/Hello-World"]}' >/dev/null

echo "triggering sync"
curl -sf -X POST "$BASE/sync"
echo

echo "dashboard:"
curl -sf "$BASE/dashboard"
echo

echo "issues payload bytes:"
curl -sf "$BASE/issues" | wc -c

echo "action-center:"
curl -sf "$BASE/action-center"
echo

echo "insights:"
curl -sf "$BASE/insights"
echo

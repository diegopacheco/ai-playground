#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
./start.sh
health=$(curl -fsS http://127.0.0.1:4177/health)
page=$(curl -fsS http://127.0.0.1:4177/)
join=$(curl -fsS -X POST -H "Content-Type: application/json" -d '{"playerId":"test-browser","name":"Test Player"}' http://127.0.0.1:4177/api/join)
echo "$health" | grep -q '"total":315'
echo "$health" | grep -q '"levels":9'
echo "$page" | grep -q "Big Cat Jigsaw"
echo "$join" | grep -q '"name":"Test Player"'
echo "Health check: $health"
echo "Browser join and nine-level 315-piece room verified"

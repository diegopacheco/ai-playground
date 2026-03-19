#!/bin/bash
echo "=== Available Agents ==="
curl -s http://localhost:3000/api/agents | python3 -m json.tool

echo ""
echo "=== Creating Auction ==="
RESULT=$(curl -s -X POST http://localhost:3000/api/auctions \
  -H "Content-Type: application/json" \
  -d '{"agents":[{"name":"claude","model":"opus","budget":100},{"name":"gemini","model":"gemini-2.5-pro","budget":100},{"name":"copilot","model":"claude-sonnet-4","budget":100}]}')
echo "$RESULT" | python3 -m json.tool

AUCTION_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo ""
echo "=== Auction ID: $AUCTION_ID ==="
echo "=== Stream: http://localhost:3000/api/auctions/$AUCTION_ID/stream ==="

echo ""
echo "=== List Auctions ==="
curl -s http://localhost:3000/api/auctions | python3 -m json.tool

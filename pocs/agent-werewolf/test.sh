#!/bin/bash
echo "=== Testing Agent Werewolf API ==="
echo ""
echo "--- List Agents ---"
curl -s http://localhost:3000/api/agents | python3 -m json.tool
echo ""
echo "--- List Games ---"
curl -s http://localhost:3000/api/games | python3 -m json.tool
echo ""
echo "--- Create Game ---"
GAME=$(curl -s -X POST http://localhost:3000/api/games \
  -H "Content-Type: application/json" \
  -d '{"agents":[{"name":"claude","model":"sonnet"},{"name":"gemini","model":"gemini-2.5-flash"},{"name":"copilot","model":"gpt-4o"},{"name":"codex","model":"o4-mini"}]}')
echo "$GAME" | python3 -m json.tool
GAME_ID=$(echo "$GAME" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo ""
echo "Game ID: $GAME_ID"
echo "Stream: http://localhost:3000/api/games/$GAME_ID/stream"
echo "Frontend: http://localhost:3001/game/$GAME_ID"

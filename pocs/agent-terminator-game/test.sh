#!/bin/bash
echo "=== Testing Agents Endpoint ==="
curl -s http://localhost:8080/api/agents | head -200
echo ""
echo "=== Testing Create Game ==="
curl -s -X POST http://localhost:8080/api/games \
  -H "Content-Type: application/json" \
  -d '{"terminator_agent":"claude","terminator_model":"sonnet","mosquito_agent":"gemini","mosquito_model":"gemini-3-flash","grid_size":"20"}'
echo ""
echo "=== Testing List Games ==="
curl -s http://localhost:8080/api/games | head -200
echo ""
echo "=== Done ==="

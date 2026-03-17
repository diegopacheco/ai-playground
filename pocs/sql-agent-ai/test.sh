#!/bin/bash
echo "Testing backend health..."
curl -s http://localhost:3000/api/schema | head -c 200
echo ""

echo ""
echo "Testing query endpoint..."
RESPONSE=$(curl -s -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Give me the top 5 salesmen by total sales"}')
echo "Response: $RESPONSE"

echo ""
echo "Testing blocked query (INSERT)..."
RESPONSE=$(curl -s -X POST http://localhost:3000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"INSERT a new salesman named Test"}')
echo "Response: $RESPONSE"

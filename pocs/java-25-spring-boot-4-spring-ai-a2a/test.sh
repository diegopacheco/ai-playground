#!/bin/bash
echo "=== Weather Agent Card ==="
curl -s http://localhost:10001/a2a/card

echo ""
echo "=== Hotel Agent Card ==="
curl -s http://localhost:10002/a2a/card

echo ""
echo "=== Trip Planner Agent Card ==="
curl -s http://localhost:10000/a2a/card

echo ""
echo "=== A2A call: Weather in Paris ==="
curl -s -X POST http://localhost:10001/a2a/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"1","params":{"message":{"role":"user","kind":"message","messageId":"msg-1","parts":[{"kind":"text","text":"What is the current weather and 7-day forecast for Paris, France?"}]}}}'

echo ""
echo "=== A2A call: Hotels in Rome ==="
curl -s -X POST http://localhost:10002/a2a/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"2","params":{"message":{"role":"user","kind":"message","messageId":"msg-1","parts":[{"kind":"text","text":"Find mid-range hotels in Rome, Italy"}]}}}'

echo ""
echo "=== A2A call: Trip Planner orchestrates weather + hotel agents ==="
curl -s -X POST http://localhost:10000/a2a/ \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"message/send","id":"3","params":{"message":{"role":"user","kind":"message","messageId":"msg-1","parts":[{"kind":"text","text":"Plan a 5-day trip to Barcelona, Spain with mid-range budget"}]}}}'

echo ""
echo "=== REST call: Trip Planner direct endpoint ==="
curl -s -X POST http://localhost:10000/a2a/plan \
  -H "Content-Type: application/json" \
  -d '{"destination":"Tokyo, Japan","days":7,"budget":"mid-range"}'

echo ""
echo "Done"

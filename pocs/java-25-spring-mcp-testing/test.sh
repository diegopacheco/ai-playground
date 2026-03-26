#!/bin/bash

mvn clean spring-boot:run --quiet >/dev/null 2>&1 &
PID=$!

for i in $(seq 1 10); do
  curl -s -o /dev/null http://localhost:8080/mcp 2>/dev/null && break
  sleep 1
done

ACCEPT="text/event-stream, application/json"
CT="application/json"
URL="http://localhost:8080/mcp"

parse_sse() {
  grep "^data:" | sed 's/^data://' | jq .
}

echo "=== INITIALIZE ==="
curl -s -D /tmp/mcp_headers -X POST $URL \
  -H "Content-Type: $CT" \
  -H "Accept: $ACCEPT" \
  -d '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl-test","version":"1.0"}},"id":1}' 2>&1 | jq .

SESSION=$(grep -i "mcp-session-id" /tmp/mcp_headers | awk '{print $2}' | tr -d '\r')
echo "Session: $SESSION"
echo ""

curl -s -X POST $URL \
  -H "Content-Type: $CT" \
  -H "Accept: $ACCEPT" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}' >/dev/null 2>&1

echo "=== LIST TOOLS ==="
curl -s -X POST $URL \
  -H "Content-Type: $CT" \
  -H "Accept: $ACCEPT" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}' 2>&1 | parse_sse
echo ""

echo "=== CALL TOOL (USD) ==="
curl -s -X POST $URL \
  -H "Content-Type: $CT" \
  -H "Accept: $ACCEPT" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"getExchangeRate","arguments":{"base":"USD"}},"id":3}' 2>&1 | parse_sse
echo ""

echo "=== CALL TOOL (EUR) ==="
curl -s -X POST $URL \
  -H "Content-Type: $CT" \
  -H "Accept: $ACCEPT" \
  -H "Mcp-Session-Id: $SESSION" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"getExchangeRate","arguments":{"base":"EUR"}},"id":4}' 2>&1 | parse_sse
echo ""

kill $PID 2>/dev/null
wait $PID 2>/dev/null

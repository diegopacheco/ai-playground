#!/bin/bash
echo "=== Testing Agent Observability Stack ==="
echo ""
echo "1. Checking Jaeger..."
JAEGER=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:16686)
if [ "$JAEGER" = "200" ]; then
    echo "   Jaeger UI is running on http://localhost:16686"
else
    echo "   Jaeger is NOT running"
fi
echo ""
echo "2. Checking Backend..."
BACKEND=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/traces)
if [ "$BACKEND" = "200" ]; then
    echo "   Backend is running on http://localhost:3000"
else
    echo "   Backend is NOT running"
fi
echo ""
echo "3. Running agent..."
RESULT=$(curl -s -X POST http://localhost:3000/api/agent/run \
    -H "Content-Type: application/json" \
    -d '{"topic":"What is observability in distributed systems?","agent":"claude"}')
echo "   Response: $RESULT"
TRACE_ID=$(echo "$RESULT" | grep -o '"trace_id":"[^"]*"' | cut -d'"' -f4)
echo "   Trace ID: $TRACE_ID"
echo ""
echo "4. Checking Frontend..."
FRONTEND=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5173)
if [ "$FRONTEND" = "200" ]; then
    echo "   Frontend is running on http://localhost:5173"
else
    echo "   Frontend is NOT running"
fi
echo ""
echo "=== Done ==="

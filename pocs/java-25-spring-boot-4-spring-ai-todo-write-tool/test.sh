#!/bin/bash

echo "=== Testing root endpoint ==="
curl -s http://localhost:8081/

echo ""
echo ""
echo "=== Testing TodoWrite with complex multi-step task ==="
curl -s -X POST http://localhost:8081/agent/ask \
  -H "Content-Type: text/plain" \
  -d "List the top 3 Spring Boot features, explain each one, give a code snippet for each, and write a summary. Use TodoWrite to track your tasks."

echo ""
echo ""
echo "=== Testing endpoint (TodoWrite auto-triggered) ==="
curl -s http://localhost:8081/agent/demo

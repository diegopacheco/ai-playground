#!/bin/bash
echo "Testing Task Subagents Orchestration..."
echo ""

echo "1. Health check:"
curl -s http://localhost:8081/actuator/health
echo ""
echo ""

echo "2. Orchestrate (complex task - architect + builder):"
curl -s -X POST http://localhost:8081/subagent/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"task":"Design and implement a Java 25 record that represents a Product with name, price, and category. Include validation in the compact constructor.","data":"Requirements: name must not be blank, price must be positive, category must be one of: ELECTRONICS, CLOTHING, FOOD"}'
echo ""
echo ""

echo "3. Orchestrate (simple task - builder direct):"
curl -s -X POST http://localhost:8081/subagent/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"task":"Write a Java method that sums a list of integers using streams.","data":""}'
echo ""

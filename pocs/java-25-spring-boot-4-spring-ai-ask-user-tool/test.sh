#!/bin/bash
curl -s -X POST http://localhost:8080/plan \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to plan a trip. Please help me plan the perfect vacation."}' | cat

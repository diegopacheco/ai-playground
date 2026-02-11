#!/bin/bash
echo "Testing user registration..."
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@test.com","password":"password123"}'
echo -e "\n"
echo "Testing user login..."
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"password123"}' | grep -o '"token":"[^"]*"' | cut -d'"' -f4)
echo "Token: $TOKEN"
echo -e "\n"
echo "Testing tweet creation..."
curl -X POST http://localhost:8000/api/tweets \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content":"Hello Twitter Clone!"}'
echo -e "\n"
echo "Testing get feed..."
curl -X GET http://localhost:8000/api/tweets/feed \
  -H "Authorization: Bearer $TOKEN"
echo -e "\n"

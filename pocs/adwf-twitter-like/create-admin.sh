#!/bin/bash

echo "Creating default admin user..."

curl -s http://localhost:8000/api/auth/register -X POST \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@twitter.local","password":"admin123"}' | jq .

echo ""
echo "Admin credentials:"
echo "  Username: admin"
echo "  Password: admin123"
echo "  Email: admin@twitter.local"

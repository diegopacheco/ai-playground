#!/bin/bash
cd backend && cargo run &
BACKEND_PID=$!
cd frontend && bun run dev &
FRONTEND_PID=$!

while ! curl -s http://localhost:3000/api/posts > /dev/null 2>&1; do
  sleep 1
done

curl -s -X POST http://localhost:3000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@twitter.com","password":"admin123"}' > /dev/null 2>&1

echo ""
echo "==================================================="
echo "  Backend running on  http://localhost:3000"
echo "  Frontend running on http://localhost:5173"
echo "==================================================="
echo "  Default user: admin@twitter.com"
echo "  Default pass: admin123"
echo "==================================================="
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait

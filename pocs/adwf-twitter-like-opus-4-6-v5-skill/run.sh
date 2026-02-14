#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

kill $(lsof -ti:8080) 2>/dev/null
kill $(lsof -ti:5173) 2>/dev/null

echo "Starting database..."
"$SCRIPT_DIR/db/start-db.sh"
echo "Applying schema..."
"$SCRIPT_DIR/db/create-schema.sh"

echo "Starting backend..."
cd "$SCRIPT_DIR/backend"
DATABASE_URL="postgresql://twitter:twitter123@localhost:5432/twitter" cargo run &
BACKEND_PID=$!

while ! curl -s http://localhost:8080/api/users/1 > /dev/null 2>&1; do
  sleep 1
done
echo "Backend is running on http://localhost:8080"

echo "Starting frontend..."
cd "$SCRIPT_DIR/frontend"
bun run dev &
FRONTEND_PID=$!

while ! curl -s http://localhost:5173 > /dev/null 2>&1; do
  sleep 1
done
echo "Frontend is running on http://localhost:5173"

echo ""
echo "=== Application Started ==="
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8080"
echo "Database: postgresql://twitter:twitter123@localhost:5432/twitter"
echo ""
echo "Default credentials:"
echo "  Email: admin@twitter.local"
echo "  Password: admin123"
echo ""
echo "Press Ctrl+C to stop all services."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; $SCRIPT_DIR/db/stop-db.sh" EXIT
wait

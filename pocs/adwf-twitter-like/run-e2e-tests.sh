#!/bin/bash

set -e

BACKEND_PID=""
FRONTEND_PID=""
DB_STARTED=0

cleanup() {
  echo "Cleaning up processes..."
  if [ ! -z "$BACKEND_PID" ]; then
    kill $BACKEND_PID 2>/dev/null || true
  fi
  if [ ! -z "$FRONTEND_PID" ]; then
    kill $FRONTEND_PID 2>/dev/null || true
  fi
  if [ $DB_STARTED -eq 1 ]; then
    echo "Stopping database..."
    ./stop-db.sh 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "======================================"
echo "Running E2E Tests for Twitter Clone"
echo "======================================"

echo "Stopping any existing services..."
./stop.sh 2>/dev/null || true
pkill -f "twitter-clone|vite" 2>/dev/null || true
lsof -ti:8000,5173 | xargs kill -9 2>/dev/null || true
sleep 2

if [ ! -f .env ]; then
  echo "Creating .env file..."
  cat > .env << 'EOF'
DATABASE_URL=postgres://twitter_user:twitter_pass@localhost/twitter_db
JWT_SECRET=your-secret-key-change-this-in-production
RUST_LOG=twitter_clone=info
EOF
fi

source .env

echo "Starting database..."
./start-db.sh
DB_STARTED=1

echo "Waiting for database to be ready..."
sleep 2

echo "Starting backend server..."
export DATABASE_URL
export JWT_SECRET
export RUST_LOG
cargo run --release > backend-test.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8000/api/auth/login -X POST -H "Content-Type: application/json" -d '{"username":"test","password":"test"}' > /dev/null 2>&1; then
    echo "Backend is ready!"
    break
  fi
  sleep 1
done

echo "Starting frontend server..."
bun run dev > frontend-test.log 2>&1 &
FRONTEND_PID=$!

echo "Waiting for frontend to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "Frontend is ready!"
    break
  fi
  sleep 1
done

echo ""
echo "======================================"
echo "Running Playwright tests..."
echo "======================================"
npx playwright test

echo ""
echo "======================================"
echo "Tests completed!"
echo "======================================"

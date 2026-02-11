#!/bin/bash

set -e

echo "Starting Twitter Clone..."

if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << 'EOF'
DATABASE_URL=postgres://twitter_user:twitter_pass@localhost/twitter_db
JWT_SECRET=your-secret-key-change-this-in-production
RUST_LOG=twitter_clone=debug,tower_http=debug
EOF
fi

source .env

echo "Starting database..."
./start-db.sh

echo "Waiting for database to be ready..."
sleep 2

echo "Starting backend server in background..."
export DATABASE_URL
export JWT_SECRET
export RUST_LOG
cargo run --release > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

echo "Waiting for backend to be ready..."
sleep 3

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend failed to start! Check backend.log"
    cat backend.log
    exit 1
fi

echo "Creating default admin user..."
curl -s http://localhost:8000/api/auth/register -X POST \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@twitter.local","password":"admin123"}' > /dev/null 2>&1 || true

echo ""
echo "==================================="
echo "Twitter Clone is starting!"
echo "==================================="
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Default Login:"
echo "  Username: admin"
echo "  Password: admin123"
echo "==================================="
echo ""

echo "Starting frontend..."
bun run dev

#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT/backend"
touch "$ROOT/backend/data.db"
SQLX_OFFLINE=true DATABASE_URL="sqlite:$ROOT/backend/data.db" cargo build --release 2>&1
if [ $? -ne 0 ]; then
    echo "Backend build failed"
    exit 1
fi

DATABASE_URL="sqlite:$ROOT/backend/data.db" ./target/release/twitter-clone-backend &
BACKEND_PID=$!
echo $BACKEND_PID > "$ROOT/.backend.pid"
echo "Backend started (pid $BACKEND_PID) on http://127.0.0.1:8080"

cd "$ROOT/frontend"
npm install --silent 2>&1
npm run dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$ROOT/.frontend.pid"
echo "Frontend started (pid $FRONTEND_PID) on http://localhost:3000"

echo "App running. Run stop.sh to stop."
wait

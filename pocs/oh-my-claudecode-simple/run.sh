#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR/backend"
cargo build --release 2>&1
./target/release/twitter-backend &
echo $! > "$SCRIPT_DIR/backend.pid"

cd "$SCRIPT_DIR/frontend"
npm install 2>&1
npm run dev &
echo $! > "$SCRIPT_DIR/frontend.pid"

echo "Backend running on http://localhost:8080"
echo "Frontend running on http://localhost:5173"

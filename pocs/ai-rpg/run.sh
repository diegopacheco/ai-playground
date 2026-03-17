#!/bin/bash
BASEDIR=$(cd "$(dirname "$0")" && pwd)
pkill -f ai-rpg-backend 2>/dev/null
pkill -f "vite" 2>/dev/null
lsof -ti:3000 | xargs kill 2>/dev/null
sleep 1
cd "$BASEDIR/backend" && cargo build --release
"$BASEDIR/backend/target/release/ai-rpg-backend" &
echo $! > "$BASEDIR/.backend.pid"
cd "$BASEDIR/frontend" && bun run dev &
echo $! > "$BASEDIR/.frontend.pid"
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:5173"

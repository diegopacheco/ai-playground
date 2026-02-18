#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$ROOT/.backend.pid" ]; then
    BACKEND_PID=$(cat "$ROOT/.backend.pid")
    kill "$BACKEND_PID" 2>/dev/null && echo "Backend stopped (pid $BACKEND_PID)"
    rm -f "$ROOT/.backend.pid"
fi

if [ -f "$ROOT/.frontend.pid" ]; then
    FRONTEND_PID=$(cat "$ROOT/.frontend.pid")
    kill "$FRONTEND_PID" 2>/dev/null && echo "Frontend stopped (pid $FRONTEND_PID)"
    rm -f "$ROOT/.frontend.pid"
fi

pkill -f "twitter-clone-backend" 2>/dev/null
pkill -f "vite" 2>/dev/null

echo "All processes stopped."

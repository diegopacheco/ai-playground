#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/.backend.pid" ]; then
    BACKEND_PID=$(cat "$SCRIPT_DIR/.backend.pid")
    kill "$BACKEND_PID" 2>/dev/null || true
    rm "$SCRIPT_DIR/.backend.pid"
    echo "Stopped backend (PID: $BACKEND_PID)"
fi
if [ -f "$SCRIPT_DIR/.frontend.pid" ]; then
    FRONTEND_PID=$(cat "$SCRIPT_DIR/.frontend.pid")
    kill "$FRONTEND_PID" 2>/dev/null || true
    rm "$SCRIPT_DIR/.frontend.pid"
    echo "Stopped frontend (PID: $FRONTEND_PID)"
fi
pkill -f "connect-four-backend" 2>/dev/null || true
pkill -f "vite preview" 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
echo "All processes stopped"

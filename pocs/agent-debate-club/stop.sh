#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$BASE_DIR/.backend.pid" ]; then
    kill $(cat "$BASE_DIR/.backend.pid") 2>/dev/null
    rm "$BASE_DIR/.backend.pid"
fi
if [ -f "$BASE_DIR/.frontend.pid" ]; then
    kill $(cat "$BASE_DIR/.frontend.pid") 2>/dev/null
    rm "$BASE_DIR/.frontend.pid"
fi
pkill -f "debate-club-backend" 2>/dev/null
pkill -f "vite" 2>/dev/null
echo "Stopped all services"

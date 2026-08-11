#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/frontend.pid"
LOG_FILE="$PROJECT_ROOT/frontend.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Frontend is already running (PID: $PID)"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

echo "Starting frontend..."

if [ ! -d "$PROJECT_ROOT/node_modules" ]; then
    echo "Installing dependencies with bun..."
    cd "$PROJECT_ROOT"
    bun install --frozen-lockfile 2>&1 || bun install 2>&1
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
fi

cd "$PROJECT_ROOT"
bun run dev > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 3

if kill -0 "$PID" 2>/dev/null; then
    echo "Frontend started successfully (PID: $PID)"
    echo "URL: http://localhost:5173"
else
    echo "Frontend failed to start. Check $LOG_FILE"
    cat "$LOG_FILE"
    exit 1
fi

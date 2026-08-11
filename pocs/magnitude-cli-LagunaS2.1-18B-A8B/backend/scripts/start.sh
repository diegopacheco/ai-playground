#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/backend.pid"
LOG_FILE="$PROJECT_ROOT/backend.log"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Backend is already running (PID: $PID)"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

echo "Starting backend..."
cd "$PROJECT_ROOT"

PIP_OUTPUT=$(pip3 install -r requirements.txt 2>&1)
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies:"
    echo "$PIP_OUTPUT"
    exit 1
fi

python3 run.py > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

sleep 2

if kill -0 "$PID" 2>/dev/null; then
    echo "Backend started successfully (PID: $PID)"
    echo "API: http://localhost:8000/api"
    echo "Health: http://localhost:8000/api/health"
else
    echo "Backend failed to start. Check $LOG_FILE"
    cat "$LOG_FILE"
    exit 1
fi

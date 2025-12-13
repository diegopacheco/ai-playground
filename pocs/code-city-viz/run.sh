#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: ./run.sh <github_url>"
    echo "   ./run.sh https://github.com/user/repo"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python3 analyze.py "$1"
if [ $? -eq 0 ]; then
    PORT=8080
    while lsof -i :$PORT > /dev/null 2>&1; do
        PORT=$((PORT + 1))
    done
    echo "Starting server on http://localhost:$PORT"
    python3 -m http.server $PORT
fi

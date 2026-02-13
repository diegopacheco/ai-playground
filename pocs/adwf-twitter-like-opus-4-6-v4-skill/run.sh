#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$SCRIPT_DIR/backend/twitter.db" ]; then
  sqlite3 "$SCRIPT_DIR/backend/twitter.db" < "$SCRIPT_DIR/db/schema.sql"
  echo "Database created."
fi

cd "$SCRIPT_DIR/backend" && cargo run &
BACKEND_PID=$!

cd "$SCRIPT_DIR/frontend" && bun run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

echo "Backend:  http://localhost:8080"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop."

wait

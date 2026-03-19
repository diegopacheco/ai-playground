#!/bin/bash
DIR=$(cd "$(dirname "$0")" && pwd)
(cd "$DIR/backend" && mvn clean package -q -DskipTests && java -jar target/terminator-game-1.0.0.jar) &
BACKEND_PID=$!
echo "$BACKEND_PID" > /tmp/terminator-game-backend.pid
(cd "$DIR/frontend" && bun install && bun run dev) &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > /tmp/terminator-game-frontend.pid
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Backend: http://localhost:8080"
echo "Frontend: http://localhost:5173"
wait

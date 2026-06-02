#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$DIR/.run"
mkdir -p "$RUN"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "WARNING: OPENAI_API_KEY is not set. The dashboard runs, but Call LLM to self-tune will be disabled."
fi

echo "Building backend jar"
(cd "$DIR/backend" && ./mvnw -q -DskipTests package)
JAR="$(ls "$DIR/backend/target/"*.jar | head -1)"

echo "Starting backend on 8080"
java -jar "$JAR" >"$RUN/backend.log" 2>&1 &
echo $! >"$RUN/backend.pid"

echo "Waiting for backend"
for i in $(seq 1 60); do
  CODE="$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/api/tune/status || true)"
  if [ "$CODE" = "200" ]; then break; fi
  sleep 1
done

if [ ! -d "$DIR/frontend/node_modules" ]; then
  echo "Installing frontend dependencies"
  (cd "$DIR/frontend" && npm install)
fi

echo "Starting frontend on 5173"
(cd "$DIR/frontend" && npm run dev) >"$RUN/frontend.log" 2>&1 &
echo $! >"$RUN/frontend.pid"

echo "Waiting for frontend"
for i in $(seq 1 60); do
  CODE="$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5173 || true)"
  if [ "$CODE" = "200" ]; then break; fi
  sleep 1
done

echo "Backend:  http://localhost:8080"
echo "Frontend: http://localhost:5173"
echo "Logs in $RUN. Stop with ./stop.sh"

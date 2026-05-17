#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
mkdir -p .run

(cd backend && mvn -q spring-boot:run > ../.run/backend.log 2>&1) &
echo $! > .run/backend.pid

(cd frontend && bun install --silent && bun run dev > ../.run/frontend.log 2>&1) &
echo $! > .run/frontend.pid

echo "backend  pid $(cat .run/backend.pid)   logs: .run/backend.log"
echo "frontend pid $(cat .run/frontend.pid)  logs: .run/frontend.log"
echo "graphiql: http://localhost:8080/graphiql"
echo "app:      http://localhost:5173"

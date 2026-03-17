#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

podman-compose down 2>/dev/null
podman-compose up -d

READY=false
for i in $(seq 1 30); do
  if podman exec sql-agent-postgres pg_isready -U sqlagent -d salesdb > /dev/null 2>&1; then
    READY=true
    break
  fi
  sleep 1
done

if [ "$READY" = false ]; then
  echo "PostgreSQL failed to start"
  exit 1
fi
echo "PostgreSQL is ready"

cd "$SCRIPT_DIR/backend"
cargo build --release 2>&1
./target/release/sql-agent-backend &
echo $! > "$SCRIPT_DIR/.backend.pid"
echo "Backend started on http://localhost:3000"

cd "$SCRIPT_DIR/frontend"
bun run dev &
echo $! > "$SCRIPT_DIR/.frontend.pid"
echo "Frontend started on http://localhost:5173"

echo ""
echo "SQL Agent is running"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:3000"
echo "  Postgres: localhost:5432"
echo ""
echo "Run ./stop.sh to stop all services"

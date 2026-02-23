#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== GraphQL-Postgres MCP Test ==="

echo ""
echo "--- Checking PostgreSQL container ---"
if ! podman exec graphmcp-postgres pg_isready -U graphmcp -d graphmcpdb > /dev/null 2>&1; then
  echo "ERROR: PostgreSQL container is not running. Run ./start.sh first."
  exit 1
fi
echo "PostgreSQL is running."

echo ""
echo "--- Checking tables via psql ---"
podman exec graphmcp-postgres psql -U graphmcp -d graphmcpdb -c "\dt"

echo ""
echo "--- Checking build ---"
if [ ! -f "$PROJECT_DIR/dist/index.js" ]; then
  echo "ERROR: dist/index.js not found. Run npm run build first."
  exit 1
fi
echo "Build OK."

echo ""
echo "--- Testing MCP server starts ---"
timeout 5 node "$PROJECT_DIR/dist/index.js" &
MCP_PID=$!
sleep 2
if kill -0 $MCP_PID 2>/dev/null; then
  echo "MCP server started successfully (PID: $MCP_PID)."
  kill $MCP_PID 2>/dev/null
else
  echo "ERROR: MCP server failed to start."
  exit 1
fi

echo ""
echo "--- Querying sample data ---"
podman exec graphmcp-postgres psql -U graphmcp -d graphmcpdb -c "SELECT * FROM users;"
podman exec graphmcp-postgres psql -U graphmcp -d graphmcpdb -c "SELECT * FROM posts;"
podman exec graphmcp-postgres psql -U graphmcp -d graphmcpdb -c "SELECT * FROM tags;"

echo ""
echo "=== All tests passed ==="

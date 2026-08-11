#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  Starting Modular CRUD Application"
echo "========================================"
echo ""

echo "→ Starting backend (Flask)..."
bash "$PROJECT_ROOT/backend/scripts/start.sh"

echo ""
echo "→ Starting frontend (React + Vite)..."
bash "$PROJECT_ROOT/frontend/scripts/start.sh"

echo ""
echo "========================================"
echo "  All services started!"
echo "========================================"
echo ""
echo "  Backend API:  http://localhost:8000/api"
echo "  Frontend:     http://localhost:5173"
echo ""
echo "  To stop all:  bash scripts/stop-all.sh"
echo "========================================"

#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "  Stopping Modular CRUD Application"
echo "========================================"
echo ""

echo "→ Stopping frontend..."
bash "$PROJECT_ROOT/frontend/scripts/stop.sh"

echo ""
echo "→ Stopping backend..."
bash "$PROJECT_ROOT/backend/scripts/stop.sh"

echo ""
echo "========================================"
echo "  All services stopped!"
echo "========================================"

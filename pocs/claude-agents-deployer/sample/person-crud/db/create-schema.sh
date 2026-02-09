#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
sqlite3 "$PROJECT_DIR/backend/persons.db" < "$SCRIPT_DIR/schema.sql"
echo "Schema created successfully in $PROJECT_DIR/backend/persons.db"

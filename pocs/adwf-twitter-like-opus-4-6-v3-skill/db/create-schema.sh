#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_PATH="$SCRIPT_DIR/../backend/twitter.db"
SCHEMA_PATH="$SCRIPT_DIR/schema.sql"
rm -f "$DB_PATH"
sqlite3 "$DB_PATH" < "$SCHEMA_PATH"
echo "Database created at $DB_PATH"

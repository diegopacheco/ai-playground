#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ ! -f "$SCRIPT_DIR/memory_game.db" ]; then
    sqlite3 "$SCRIPT_DIR/memory_game.db" < "$SCRIPT_DIR/schema.sql"
    echo "Database created and schema applied."
else
    sqlite3 "$SCRIPT_DIR/memory_game.db" < "$SCRIPT_DIR/schema.sql"
    echo "Database exists. Schema re-applied."
fi

#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DB_PATH="$SCRIPT_DIR/../backend/twitter.db"
sqlite3 "$DB_PATH"

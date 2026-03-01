#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
sqlite3 "$SCRIPT_DIR/memory_game.db"

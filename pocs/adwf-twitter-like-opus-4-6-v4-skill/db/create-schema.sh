#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p ../backend
sqlite3 ../backend/twitter.db < schema.sql
echo "Schema applied to ../backend/twitter.db"

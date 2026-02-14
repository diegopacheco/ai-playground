#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
podman exec -i twitter-postgres psql -U twitter -d twitter < "$SCRIPT_DIR/schema.sql"
echo "Schema applied."

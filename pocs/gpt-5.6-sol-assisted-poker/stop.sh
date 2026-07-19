#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .server.pid ]; then
    echo "Riverlight is not running"
    exit 0
fi
server_pid=$(tr -d '[:space:]' < .server.pid)
if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
fi
rm -f .server.pid
echo "Riverlight stopped"


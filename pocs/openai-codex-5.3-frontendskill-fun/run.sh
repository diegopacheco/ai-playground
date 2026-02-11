#!/bin/bash
set -e
PORT="${1:-4173}"
echo "http://127.0.0.1:${PORT}/?p=1"
python3 -m http.server "$PORT"

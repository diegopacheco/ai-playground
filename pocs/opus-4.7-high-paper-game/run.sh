#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT=8080
echo "Serving at http://localhost:${PORT}"
python3 -m http.server "${PORT}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/app"

if [ ! -d node_modules ]; then
  npm install --silent
fi

exec node chat.js "$@"

#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d node_modules ]; then
  npm install
fi
npm run build
test -f dist/index.html
echo "All checks passed"

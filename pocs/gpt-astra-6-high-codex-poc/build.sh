#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for file in server.js js/*.js; do
  node --check "$file"
done
rm -rf dist
mkdir -p dist
cp index.html style.css dist/
cp -R js assets dist/
echo "Build ready in dist/"

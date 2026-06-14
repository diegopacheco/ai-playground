#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d node_modules ]; then
  echo "installing dependencies"
  npm install --no-audit --no-fund
fi

echo "starting Nimbus Commerce Console on http://localhost:5188"
npm run dev

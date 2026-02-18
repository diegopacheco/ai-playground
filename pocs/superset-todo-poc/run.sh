#!/bin/zsh
set -e

if [ ! -d node_modules ]; then
  npm install
fi

npm run dev -- --host 0.0.0.0 --port 4173

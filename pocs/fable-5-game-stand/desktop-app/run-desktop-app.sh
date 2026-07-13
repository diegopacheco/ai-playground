#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x node_modules/.bin/electron ] || [ ! -x .venv/bin/python ]; then
  echo "Run ./install.sh first"
  exit 1
fi
exec node_modules/.bin/electron .

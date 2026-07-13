#!/bin/bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required"
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required"
  exit 1
fi
python3 -m venv .venv
.venv/bin/pip install -q -r ../requirements.txt
npm install
echo "Game Stand desktop app installed"

#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  python3 -m venv venv
  ./venv/bin/pip install --quiet opencv-python numpy
fi
if [ -f app.pid ] && kill -0 "$(cat app.pid)" 2>/dev/null; then
  echo "already running with pid $(cat app.pid)"
  exit 0
fi
./venv/bin/python app.py &
echo $! > app.pid
echo "started with pid $(cat app.pid)"

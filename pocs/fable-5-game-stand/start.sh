#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python app.py &
echo $! > .server.pid
for i in $(seq 1 30); do
  if curl -s -o /dev/null http://127.0.0.1:5057/; then
    echo "Game Stand running at http://127.0.0.1:5057"
    exit 0
  fi
  sleep 1
done
echo "server did not start"
exit 1

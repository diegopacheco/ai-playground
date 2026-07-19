#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .server.pid ] && kill -0 "$(tr -d '[:space:]' < .server.pid)" 2>/dev/null; then
    echo "Riverlight is already running at http://127.0.0.1:8000"
    exit 0
fi
if [ ! -x .venv/bin/python ]; then
    python3.14 -m venv .venv
fi
.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python manage.py migrate --noinput
nohup .venv/bin/python manage.py runserver 127.0.0.1:8000 --noreload > .server.log 2>&1 &
server_pid=$!
echo "$server_pid" > .server.pid
attempt=0
until curl -fsS http://127.0.0.1:8000 >/dev/null 2>&1; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Riverlight could not start"
        tail -n 30 .server.log
        exit 1
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 60 ]; then
        echo "Riverlight did not become ready within 60 seconds"
        exit 1
    fi
    sleep 1
done
echo "Riverlight is running at http://127.0.0.1:8000"


#!/bin/bash
if [ -f app.pid ]; then
    kill "$(cat app.pid)" 2>/dev/null
    rm -f app.pid
fi
PID=$(lsof -ti:8080 2>/dev/null)
if [ -n "$PID" ]; then
    kill $PID 2>/dev/null
    echo "Stopped"
else
    echo "Nothing running on port 8080"
fi

#!/bin/bash
cd "$(dirname "$0")"
if [ -f app.pid ]; then
    kill $(cat app.pid) 2>/dev/null
    rm app.pid
    echo "Stock app stopped"
else
    echo "No PID file found"
fi

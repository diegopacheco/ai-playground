#!/bin/bash

cd "$(dirname "$0")"

if [ -f .backend.pid ]; then
    kill "$(cat .backend.pid)" 2>/dev/null
    rm -f .backend.pid
fi

if [ -f .frontend.pid ]; then
    kill "$(cat .frontend.pid)" 2>/dev/null
    rm -f .frontend.pid
fi

lsof -ti:3001 | xargs kill 2>/dev/null
lsof -ti:5173 | xargs kill 2>/dev/null

echo "Stopped all services"

#!/bin/bash
if [ -f .backend.pid ]; then
    kill $(cat .backend.pid) 2>/dev/null
    rm .backend.pid
fi
if [ -f .frontend.pid ]; then
    kill $(cat .frontend.pid) 2>/dev/null
    rm .frontend.pid
fi
pkill -f "local-agent-orama-backend" 2>/dev/null
pkill -f "vite" 2>/dev/null
echo "Services stopped"

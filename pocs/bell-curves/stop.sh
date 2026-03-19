#!/bin/bash
if [ -f /tmp/bell-curves-backend.pid ]; then
    kill $(cat /tmp/bell-curves-backend.pid) 2>/dev/null
    rm /tmp/bell-curves-backend.pid
    echo "Backend stopped"
fi
if [ -f /tmp/bell-curves-frontend.pid ]; then
    kill $(cat /tmp/bell-curves-frontend.pid) 2>/dev/null
    rm /tmp/bell-curves-frontend.pid
    echo "Frontend stopped"
fi
pkill -f "bell-curves/backend/server" 2>/dev/null
pkill -f "bell-curves/frontend" 2>/dev/null
echo "All processes stopped"

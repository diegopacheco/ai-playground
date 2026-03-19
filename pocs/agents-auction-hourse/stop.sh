#!/bin/bash
kill $(cat /tmp/auction-backend.pid 2>/dev/null) 2>/dev/null
kill $(cat /tmp/auction-frontend.pid 2>/dev/null) 2>/dev/null
pkill -f "auction-server" 2>/dev/null
pkill -f "vite.*5173" 2>/dev/null
pkill -f "vite.*5174" 2>/dev/null
lsof -ti:3000 | xargs kill 2>/dev/null
lsof -ti:5173 | xargs kill 2>/dev/null
lsof -ti:5174 | xargs kill 2>/dev/null
rm -f /tmp/auction-backend.pid /tmp/auction-frontend.pid
echo "Stopped"

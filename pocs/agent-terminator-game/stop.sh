#!/bin/bash
kill $(cat /tmp/terminator-game-backend.pid 2>/dev/null) 2>/dev/null
kill $(cat /tmp/terminator-game-frontend.pid 2>/dev/null) 2>/dev/null
pkill -f "terminator-game-1.0.0.jar" 2>/dev/null
pkill -f "remix.*5173" 2>/dev/null
lsof -ti:8080 | xargs kill 2>/dev/null
lsof -ti:5173 | xargs kill 2>/dev/null
rm -f /tmp/terminator-game-backend.pid /tmp/terminator-game-frontend.pid
echo "Stopped"

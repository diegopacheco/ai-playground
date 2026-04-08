#!/bin/bash

if [ -f /tmp/tetris-backend.pid ]; then
  kill $(cat /tmp/tetris-backend.pid) 2>/dev/null
  rm /tmp/tetris-backend.pid
fi

if [ -f /tmp/tetris-frontend.pid ]; then
  kill $(cat /tmp/tetris-frontend.pid) 2>/dev/null
  rm /tmp/tetris-frontend.pid
fi

pkill -f "tetris-backend" 2>/dev/null
pkill -f "vite.*5173" 2>/dev/null

lsof -ti:8080 | xargs kill 2>/dev/null
lsof -ti:5173 | xargs kill 2>/dev/null

echo "All Tetris processes stopped"

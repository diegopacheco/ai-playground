#!/bin/bash

if [ -f /tmp/tetris-backend.pid ]; then
  kill $(cat /tmp/tetris-backend.pid) 2>/dev/null
  rm /tmp/tetris-backend.pid
  echo "Backend stopped"
fi

if [ -f /tmp/tetris-frontend.pid ]; then
  kill $(cat /tmp/tetris-frontend.pid) 2>/dev/null
  rm /tmp/tetris-frontend.pid
  echo "Frontend stopped"
fi

pkill -f "tetris-backend" 2>/dev/null
pkill -f "vite.*5173" 2>/dev/null

echo "All Tetris processes stopped"

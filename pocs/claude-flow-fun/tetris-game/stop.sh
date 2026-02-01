#!/bin/bash
pkill -f "tetris-backend" 2>/dev/null || true
pkill -f "vite.*tetris-frontend" 2>/dev/null || true
pkill -f "node.*vite" 2>/dev/null || true
echo "Stopped Tetris game processes"

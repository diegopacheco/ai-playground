#!/bin/bash
pkill -f "memory-game-backend" 2>/dev/null
pkill -f "vite" 2>/dev/null
echo "Stopped backend and frontend"

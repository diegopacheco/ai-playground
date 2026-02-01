#!/bin/bash
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
pkill -f tetris-backend 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
echo "Stopped all tetris processes"

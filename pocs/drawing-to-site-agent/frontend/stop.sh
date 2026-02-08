#!/bin/bash
pkill -f "vite.*drawing-to-site" 2>/dev/null
pkill -f "bun.*dev" 2>/dev/null
echo "Frontend stopped"

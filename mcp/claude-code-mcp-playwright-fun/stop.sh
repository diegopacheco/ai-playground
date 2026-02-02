#!/bin/bash
pkill -f "deno run.*server.ts" 2>/dev/null
pkill -f "vite" 2>/dev/null
echo "Stopped all servers"

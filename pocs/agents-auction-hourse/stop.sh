#!/bin/bash
kill $(cat /tmp/auction-backend.pid 2>/dev/null) 2>/dev/null
kill $(cat /tmp/auction-frontend.pid 2>/dev/null) 2>/dev/null
pkill -f "auction-server" 2>/dev/null
pkill -f "vite.*agents-auction" 2>/dev/null
rm -f /tmp/auction-backend.pid /tmp/auction-frontend.pid
echo "Stopped"

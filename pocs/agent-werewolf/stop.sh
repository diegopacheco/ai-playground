#!/bin/bash
kill $(cat /tmp/werewolf-backend.pid 2>/dev/null) 2>/dev/null
kill $(cat /tmp/werewolf-frontend.pid 2>/dev/null) 2>/dev/null
pkill -f "werewolf-server" 2>/dev/null
pkill -f "next.*3001" 2>/dev/null
lsof -ti:3000 | xargs kill 2>/dev/null
lsof -ti:3001 | xargs kill 2>/dev/null
rm -f /tmp/werewolf-backend.pid /tmp/werewolf-frontend.pid
echo "Stopped"

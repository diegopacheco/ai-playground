#!/bin/bash
if [ -f backend.pid ]; then
  kill $(cat backend.pid) 2>/dev/null
  rm backend.pid
  echo "Backend stopped"
fi
if [ -f frontend.pid ]; then
  kill $(cat frontend.pid) 2>/dev/null
  rm frontend.pid
  echo "Frontend stopped"
fi
lsof -ti:8080 | xargs kill 2>/dev/null
lsof -ti:5173 | xargs kill 2>/dev/null
echo "Ports 8080 and 5173 cleared"

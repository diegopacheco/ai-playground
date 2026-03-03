#!/bin/bash
if [ -f /tmp/memory-backend.pid ]; then
  kill $(cat /tmp/memory-backend.pid) 2>/dev/null
  rm /tmp/memory-backend.pid
  echo "Backend stopped"
fi
if [ -f /tmp/memory-frontend.pid ]; then
  kill $(cat /tmp/memory-frontend.pid) 2>/dev/null
  rm /tmp/memory-frontend.pid
  echo "Frontend stopped"
fi

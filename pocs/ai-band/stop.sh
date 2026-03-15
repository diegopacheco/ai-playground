#!/bin/bash
if [ -f /tmp/ai-band-backend.pid ]; then
  kill $(cat /tmp/ai-band-backend.pid) 2>/dev/null
  rm /tmp/ai-band-backend.pid
fi
if [ -f /tmp/ai-band-frontend.pid ]; then
  kill $(cat /tmp/ai-band-frontend.pid) 2>/dev/null
  rm /tmp/ai-band-frontend.pid
fi
pkill -f "ai-band" 2>/dev/null
pkill -f "react-scripts start" 2>/dev/null
echo "AI Band stopped"

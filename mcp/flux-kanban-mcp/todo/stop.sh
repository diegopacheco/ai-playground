#!/usr/bin/env bash
if [ -f /tmp/grocery/server.pid ]; then
  kill "$(cat /tmp/grocery/server.pid)" 2>/dev/null || true
  rm /tmp/grocery/server.pid
fi
if [ -f /tmp/grocery/app.pid ]; then
  kill "$(cat /tmp/grocery/app.pid)" 2>/dev/null || true
  rm /tmp/grocery/app.pid
fi
echo "Stopped."

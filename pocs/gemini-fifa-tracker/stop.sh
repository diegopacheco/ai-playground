#!/bin/bash
if [ -f .server.pid ]; then
  PID=$(cat .server.pid)
  kill $PID
  rm .server.pid
  echo "Server with PID $PID stopped"
else
  echo "No server running"
fi

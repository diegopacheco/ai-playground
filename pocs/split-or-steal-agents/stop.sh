#!/bin/bash
cd "$(dirname "$0")"
if [ -f .server.pid ]; then
  kill "$(cat .server.pid)" 2>/dev/null
  rm -f .server.pid
fi
lsof -ti tcp:8017 | xargs kill -9 2>/dev/null
echo "Split or Steal stopped"

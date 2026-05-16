#!/usr/bin/env bash

PIDS=$(lsof -ti :3000 -ti :4000 2>/dev/null || true)

if [ -z "$PIDS" ]; then
  echo "Nothing running on ports 3000 or 4000."
  exit 0
fi

echo "Stopping PIDs: $PIDS"
kill $PIDS 2>/dev/null || true

for i in 1 2 3 4 5; do
  REMAIN=$(lsof -ti :3000 -ti :4000 2>/dev/null || true)
  [ -z "$REMAIN" ] && break
  sleep 1
done

REMAIN=$(lsof -ti :3000 -ti :4000 2>/dev/null || true)
if [ -n "$REMAIN" ]; then
  echo "Force killing: $REMAIN"
  kill -9 $REMAIN 2>/dev/null || true
fi

echo "Stopped."

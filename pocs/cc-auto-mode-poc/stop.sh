#!/usr/bin/env bash
if pgrep -f "expo start" >/dev/null 2>&1; then
  pkill -f "expo start"
  echo "Stopped Expo"
else
  echo "No Expo process running (if started with start.sh, press Ctrl+C in that terminal)"
fi

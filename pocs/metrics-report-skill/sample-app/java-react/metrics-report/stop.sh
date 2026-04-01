#!/bin/bash

PID=$(lsof -ti:3737)
if [ -n "$PID" ]; then
  kill -9 $PID
  echo "Stopped serve process on port 3737 (PID: $PID)"
else
  echo "No process found on port 3737"
fi

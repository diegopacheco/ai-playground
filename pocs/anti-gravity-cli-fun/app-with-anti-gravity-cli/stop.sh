#!/bin/bash
if [ -f dev-server.pid ]; then
  kill $(cat dev-server.pid)
  rm dev-server.pid
  echo "Server stopped."
else
  echo "No running server found."
fi

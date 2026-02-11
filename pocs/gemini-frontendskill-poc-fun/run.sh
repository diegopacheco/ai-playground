#!/bin/bash
# Start a simple python server in the background
python3 -m http.server 8000 > /dev/null 2>&1 &
PID=$!

echo "Server started at http://localhost:8000"
echo "Opening Landing Page 1..."
open "http://localhost:8000/pages/1.html"

echo "Press Ctrl+C to stop the server."

# Function to kill server on exit
trap "kill $PID" EXIT

# Keep script running
while true; do sleep 1; done

#!/bin/bash

# Rock Paper Scissors Game - Development Server
echo "Starting Rock Paper Scissors Game..."

# Start the development server
bun run dev &
SERVER_PID=$!

# Wait a moment for the server to start
sleep 2

# Open the game in the default browser
if command -v open &> /dev/null; then
    # macOS
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    # Linux
    xdg-open http://localhost:3000
elif command -v start &> /dev/null; then
    # Windows
    start http://localhost:3000
else
    echo "Please open http://localhost:3000 in your browser"
fi

echo "Game is running at http://localhost:3000"
echo "Press Ctrl+C to stop the server"

# Wait for the server process
wait $SERVER_PID
#!/bin/bash

echo "🎮 Starting Who Wants to be a Vibe Coder? 🚀"
echo ""

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
    echo ""
fi

echo "🌟 Launching the game..."
echo "🎯 The game will open at http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

npm start

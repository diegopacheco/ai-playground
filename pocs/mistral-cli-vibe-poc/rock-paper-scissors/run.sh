#!/bin/bash

# Rock Paper Scissors Game - Run Script
# This script provides easy commands to run the game in different modes

echo "🎮 Rock Paper Scissors Game"
echo "============================"
echo ""
echo "Choose an option:"
echo "1) Development mode (with hot reload)"
echo "2) Production build"
echo "3) Preview production build"
echo "4) Install dependencies"
echo ""

read -p "Enter your choice (1-4): " choice

echo ""

case $choice in
    1)
        echo "🚀 Starting development server..."
        echo "The game will be available at http://localhost:5173/"
        echo "Press Ctrl+C to stop the server"
        echo ""
        bun dev
        ;;
    2)
        echo "🔨 Building production version..."
        bun run build
        echo ""
        echo "✅ Build completed! Files are in the 'dist' folder"
        ;;
    3)
        echo "🌐 Starting production preview..."
        echo "The game will be available at http://localhost:4173/"
        echo "Press Ctrl+C to stop the server"
        echo ""
        bun run preview
        ;;
    4)
        echo "📦 Installing dependencies..."
        bun install
        echo ""
        echo "✅ Dependencies installed successfully!"
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again and choose 1-4."
        exit 1
        ;;
esac
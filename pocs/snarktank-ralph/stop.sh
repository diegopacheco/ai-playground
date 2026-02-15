#!/bin/bash

pkill -f "react-scripts start" 2>/dev/null
pkill -f "node src/index.js" 2>/dev/null
pkill -f "concurrently" 2>/dev/null

echo "SnarkTank stopped."

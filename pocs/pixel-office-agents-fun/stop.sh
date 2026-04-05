#!/bin/bash
cd "$(dirname "$0")"
echo "=== Stopping Pixel Office ==="
cd frontend && bash stop.sh && cd ..
cd backend && bash stop.sh && cd ..
echo "=== Pixel Office stopped ==="

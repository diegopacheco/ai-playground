#!/bin/bash
cd "$(dirname "$0")"
echo "=== Starting Pixel Office ==="
cd backend && bash run.sh && cd ..
cd frontend && bash run.sh && cd ..
echo "=== Pixel Office is running ==="
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:3001"

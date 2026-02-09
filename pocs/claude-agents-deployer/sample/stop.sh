#!/bin/bash
cd "$(dirname "$0")"

echo "Stopping frontend..."
if [ -f /tmp/blog-frontend.pid ]; then
    FRONTEND_PID=$(cat /tmp/blog-frontend.pid)
    kill $FRONTEND_PID 2>/dev/null
    rm /tmp/blog-frontend.pid
fi

echo "Stopping backend..."
if [ -f /tmp/blog-backend.pid ]; then
    BACKEND_PID=$(cat /tmp/blog-backend.pid)
    kill $BACKEND_PID 2>/dev/null
    rm /tmp/blog-backend.pid
fi

echo "Stopping database..."
cd backend
podman-compose down
cd ..

echo "All services stopped."

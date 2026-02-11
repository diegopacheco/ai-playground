#!/bin/bash

echo "Stopping Twitter Clone..."

echo "Killing backend server..."
pkill -9 -f twitter-clone || true
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

echo "Killing frontend dev server..."
pkill -9 -f "vite" || true
lsof -ti:5173 | xargs kill -9 2>/dev/null || true

echo "Stopping database containers..."
podman stop twitter_postgres 2>/dev/null || true
podman rm twitter_postgres 2>/dev/null || true
podman stop test_postgres_1 2>/dev/null || true
podman rm test_postgres_1 2>/dev/null || true
podman-compose down 2>/dev/null || true

echo "All services stopped!"

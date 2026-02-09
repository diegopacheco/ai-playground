#!/bin/bash
cd "$(dirname "$0")"

echo "Starting database..."
cd backend
podman-compose up -d
echo "Waiting for PostgreSQL to be ready..."
until podman exec blog-postgres pg_isready -U postgres > /dev/null 2>&1; do
    sleep 1
done
echo "PostgreSQL is ready."

echo "Starting backend..."
cargo build --release
./target/release/blog-platform &
BACKEND_PID=$!
echo $BACKEND_PID > /tmp/blog-backend.pid
cd ..

echo "Waiting for backend to be ready..."
until curl -s http://localhost:8080/api/posts > /dev/null 2>&1; do
    sleep 1
done
echo "Backend is ready."

echo "Starting frontend..."
cd frontend
npm install --silent
npm run dev &
FRONTEND_PID=$!
echo $FRONTEND_PID > /tmp/blog-frontend.pid
cd ..

echo "All services started."
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8080"

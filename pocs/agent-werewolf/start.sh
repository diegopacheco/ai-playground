#!/bin/bash
podman-compose up --build -d
echo "Waiting for services to start..."
until podman-compose ps | grep -q "Up"; do
  sleep 1
done
echo "Backend: http://localhost:3000"
echo "Frontend: http://localhost:3001"

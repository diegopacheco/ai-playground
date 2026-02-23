#!/bin/bash
podman-compose up -d
echo "Waiting for PostgreSQL to be ready..."
until podman exec graphmcp-postgres pg_isready -U graphmcp -d graphmcpdb > /dev/null 2>&1; do
  sleep 1
done
echo "PostgreSQL is ready."

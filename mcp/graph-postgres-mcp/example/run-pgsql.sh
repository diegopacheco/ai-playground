#!/bin/bash
cd "$(dirname "$0")"
podman-compose up -d
echo "Waiting for PostgreSQL to be ready..."
until podman exec graphmcp-example-postgres pg_isready -U exampleuser -d exampledb > /dev/null 2>&1; do
  sleep 1
done
echo "PostgreSQL is ready on port 5433."
echo ""
echo "Connection details:"
echo "  Host:     localhost"
echo "  Port:     5433"
echo "  User:     exampleuser"
echo "  Password: examplepass"
echo "  Database: exampledb"
echo ""
echo "Tables: salesmen, buyers, products, orders"

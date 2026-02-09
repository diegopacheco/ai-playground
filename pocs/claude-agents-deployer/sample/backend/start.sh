#!/bin/bash
podman-compose up -d
echo "Waiting for PostgreSQL to be ready..."
until podman exec blog-postgres pg_isready -U postgres > /dev/null 2>&1; do
    sleep 1
done
echo "PostgreSQL is ready."
cargo run

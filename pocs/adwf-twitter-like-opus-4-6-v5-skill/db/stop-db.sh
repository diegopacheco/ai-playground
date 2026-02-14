#!/bin/bash
podman stop twitter-postgres 2>/dev/null
podman rm twitter-postgres 2>/dev/null
echo "PostgreSQL container stopped and removed."

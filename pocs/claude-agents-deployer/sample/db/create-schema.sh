#!/bin/bash
podman exec -i blogdb-postgres psql -U bloguser -d blogdb < "$(dirname "$0")/schema.sql"

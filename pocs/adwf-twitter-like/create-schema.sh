#!/bin/bash
podman exec -i twitter_postgres psql -U twitter_user -d twitter_db < db/schema.sql
echo "Schema created successfully!"

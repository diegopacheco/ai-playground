#!/bin/bash
podman exec -it blogdb-postgres psql -U bloguser -d blogdb

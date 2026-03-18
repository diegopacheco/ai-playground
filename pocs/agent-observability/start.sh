#!/bin/bash
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
podman-compose -f "$BASE_DIR/podman-compose.yml" up -d

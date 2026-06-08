#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
podman-compose down
printf "website stopped\n"

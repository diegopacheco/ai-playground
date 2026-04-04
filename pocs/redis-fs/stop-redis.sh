#!/bin/bash
podman stop redis-fs 2>/dev/null
podman rm redis-fs 2>/dev/null
echo "Redis stopped"

#!/bin/bash
podman run -d --name redis-fs -p 6379:6379 redis:8.0.2
while ! podman exec redis-fs redis-cli ping 2>/dev/null | grep -q PONG; do
  sleep 1
done
echo "Redis is ready"

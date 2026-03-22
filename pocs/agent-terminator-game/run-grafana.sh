#!/bin/bash
podman-compose up -d
echo "Waiting for Grafana to start..."
while true; do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health 2>/dev/null)
  if [ "$STATUS" = "200" ]; then
    break
  fi
  sleep 1
done
echo "Grafana is ready at http://localhost:3000"
echo "Prometheus is ready at http://localhost:9090"
echo "Your app must be running on port 8080 with metrics at /actuator/prometheus"

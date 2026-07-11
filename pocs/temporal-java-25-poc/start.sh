#!/usr/bin/env bash
set -euo pipefail
start_service() {
  local container="$1"
  local service="$2"
  if podman container exists "$container"; then
    podman start "$container" >/dev/null
  else
    podman-compose -f compose.yml up -d "$service"
  fi
}
start_service temporal-java-25-poc_postgres_1 postgres
for i in {1..60}; do
  podman exec temporal-java-25-poc_postgres_1 pg_isready -U agents -d agents >/dev/null 2>&1 && break
  if [ "$i" -eq 60 ]; then
    printf '%s\n' 'PostgreSQL did not start'
    exit 1
  fi
  sleep 1
done
start_service temporal-java-25-poc_temporal_1 temporal
for i in {1..60}; do
  nc -z 127.0.0.1 7233 >/dev/null 2>&1 && break
  if [ "$i" -eq 60 ]; then
    printf '%s\n' 'Temporal did not start'
    exit 1
  fi
  sleep 1
done
podman exec temporal-java-25-poc_temporal_1 temporal operator namespace create --namespace default --address "$(podman inspect temporal-java-25-poc_temporal_1 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'):7233" >/dev/null 2>&1 || true
start_service temporal-java-25-poc_temporal-ui_1 temporal-ui
if lsof -ti :8082 >/dev/null 2>&1; then
  lsof -ti :8082 | xargs kill
  for i in {1..30}; do
    nc -z 127.0.0.1 8082 >/dev/null 2>&1 || break
    sleep 1
  done
fi
mvn -q clean package -DskipTests
java -jar target/temporal-java-25-poc-1.0.0.jar

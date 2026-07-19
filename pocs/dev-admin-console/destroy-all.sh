#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" != "--yes" ]; then
  echo "This destroys EVERYTHING:"
  echo "  - the backend and frontend processes"
  echo "  - the metadata postgres container AND its volume"
  echo "    (projects, connections, users, saved queries, the whole audit trail)"
  echo "  - every demo target container and its data"
  echo "  - built artifacts (backend/target, frontend/dist, frontend/node_modules/.vite)"
  echo ""
  printf 'type DESTROY to continue: '
  read -r answer
  if [ "$answer" != "DESTROY" ]; then
    echo "aborted, nothing was removed"
    exit 1
  fi
fi

./frontend/stop.sh > /dev/null 2>&1 || true
./backend/stop.sh > /dev/null 2>&1 || true
pkill -f "storybook dev" > /dev/null 2>&1 || true

if podman info > /dev/null 2>&1; then
  for container in admin-console-metadata \
                   admin-console-demo-mysql admin-console-demo-postgres \
                   admin-console-demo-cassandra admin-console-demo-redis \
                   admin-console-demo-etcd admin-console-demo-kafka \
                   admin-console-demo-elasticsearch; do
    podman rm -f "$container" > /dev/null 2>&1 || true
  done

  podman volume ls --format '{{.Name}}' 2>/dev/null | while IFS= read -r volume; do
    case "$volume" in
      *admin-console*|*admin_console*|demo_*) podman volume rm -f "$volume" > /dev/null 2>&1 || true ;;
    esac
  done

  for network in admin-console_default demo_default; do
    podman network rm "$network" > /dev/null 2>&1 || true
  done
fi

rm -rf backend/target frontend/dist frontend/node_modules/.vite frontend/.astro
rm -f /tmp/admin-console-backend.pid /tmp/admin-console-frontend.pid /tmp/admin-console-demo-cookies.txt

echo "everything destroyed — run ./start.sh to begin from scratch"

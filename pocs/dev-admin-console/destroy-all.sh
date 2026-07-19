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
  for container in dev-admin-console-metadata \
                   dev-admin-console-demo-mysql dev-admin-console-demo-postgres \
                   dev-admin-console-demo-cassandra dev-admin-console-demo-redis \
                   dev-admin-console-demo-etcd dev-admin-console-demo-kafka \
                   dev-admin-console-demo-elasticsearch; do
    podman rm -f "$container" > /dev/null 2>&1 || true
  done

  podman volume ls --format '{{.Name}}' 2>/dev/null | while IFS= read -r volume; do
    case "$volume" in
      *dev-admin-console*|*dev_admin_console*|demo_*) podman volume rm -f "$volume" > /dev/null 2>&1 || true ;;
    esac
  done

  for network in dev-admin-console_default demo_default; do
    podman network rm "$network" > /dev/null 2>&1 || true
  done
fi

rm -rf backend/target frontend/dist frontend/node_modules/.vite frontend/.astro
rm -f /tmp/dev-admin-console-backend.pid /tmp/dev-admin-console-frontend.pid /tmp/dev-admin-console-demo-cookies.txt

echo "everything destroyed — run ./start.sh to begin from scratch"

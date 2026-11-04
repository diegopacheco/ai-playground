#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

podman container exists agentinfra-postgres || fail "postgres is not running, run scripts/start-all.sh"
if [ -t 0 ]; then
  exec podman exec -it agentinfra-postgres psql -U agentinfra -d agentinfra "$@"
fi
exec podman exec -i agentinfra-postgres psql -U agentinfra -d agentinfra "$@"

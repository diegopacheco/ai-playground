#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
RUN="$ROOT/.run"
LOGS="$RUN/logs"
cd "$ROOT"

mkdir -p "$RUN" "$LOGS"

SERVICES=""
if [ -f "$SCRIPTS/ports.env" ]; then
  SERVICES="$(grep -v '^[[:space:]]*#' "$SCRIPTS/ports.env" | grep '=' || true)"
fi

service_names() {
  printf "%s\n" "$SERVICES" | sed '/^$/d' | cut -d= -f1
}

service_port() {
  printf "%s\n" "$SERVICES" | sed '/^$/d' | awk -F= -v n="$1" '$1==n{print $2; exit}'
}

service_url() {
  local port
  port="$(service_port "$1")"
  case "$1" in
    postgres) printf "postgresql://localhost:%s\n" "$port" ;;
    mysql|mariadb) printf "mysql://localhost:%s\n" "$port" ;;
    redis|valkey) printf "redis://localhost:%s\n" "$port" ;;
    mongodb) printf "mongodb://localhost:%s\n" "$port" ;;
    *) printf "http://localhost:%s\n" "$port" ;;
  esac
}

port_pid() {
  lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null | head -1 || true
}

port_up() {
  [ -n "$(port_pid "$1")" ]
}

wait_port_up() {
  local tries
  tries="${2:-60}"
  while [ "$tries" -gt 0 ]; do
    if port_up "$1"; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

wait_port_down() {
  local tries
  tries="${2:-30}"
  while [ "$tries" -gt 0 ]; do
    if ! port_up "$1"; then return 0; fi
    sleep 1
    tries=$((tries - 1))
  done
  return 1
}

start_bg() {
  local name dir
  name="$1"
  dir="$2"
  shift 2
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
    log "$name already running"
    return 0
  fi
  ( cd "$dir" && exec "$@" >"$LOGS/$name.log" 2>&1 ) &
  echo $! >"$RUN/$name.pid"
  log "$name started pid $!"
}

stop_bg() {
  local name pid port
  name="$1"
  if [ -f "$RUN/$name.pid" ]; then
    pid="$(cat "$RUN/$name.pid")"
    if kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" 2>/dev/null || true
      kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$RUN/$name.pid"
  fi
  port="$(service_port "$name" || true)"
  if [ -n "$port" ]; then
    wait_port_down "$port" 10 || true
    pid="$(port_pid "$port")"
    if [ -n "$pid" ]; then kill -KILL "$pid" 2>/dev/null || true; fi
  fi
  log "$name stopped"
}

open_url() {
  if command -v open >/dev/null 2>&1; then
    open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$1"
  else
    fail "no browser opener found, open $1 manually"
  fi
}

require() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required but not installed"
}

log() {
  printf "%s\n" "$*"
}

fail() {
  printf "ERROR: %s\n" "$*" >&2
  exit 1
}

CLUSTER_ENV="$SCRIPTS/cluster.env"
. "$CLUSTER_ENV"
SUBSTRATE="$ROOT/.substrate"
BIN="$ROOT/.bin"
SHIM="$RUN/shim"
KCTX="kind-$KIND_CLUSTER_NAME"
mkdir -p "$BIN" "$SHIM"

ate() {
  "$BIN/kubectl-ate" --context "$KCTX" "$@" 2> >(grep -v "no --token-file" >&2)
}

cpuset_delegated() {
  podman machine ssh "cat /sys/fs/cgroup/user.slice/user-\$(id -u).slice/user@\$(id -u).service/cgroup.controllers" | grep -qw cpuset
}

kube() {
  kubectl --context "$KCTX" "$@"
}

ko_bin() {
  (cd "$SUBSTRATE/hack/tools/ko" && go tool -n ko)
}

node_platform() {
  printf "linux/%s\n" "$(podman info --format '{{.Host.Arch}}')"
}

podman_env() {
  cat >"$SHIM/docker" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [ "${1:-}" = "buildx" ] && [ "${2:-}" = "build" ]; then
  shift 2
  meta=""; tag=""; args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --push) shift ;;
      --metadata-file) meta="$2"; shift 2 ;;
      --platform=*) args+=("--platform=linux/$(podman info --format '{{.Host.Arch}}')"); shift ;;
      --cache-from|--cache-to) shift 2 ;;
      -t) tag="$2"; args+=("-t" "$2"); shift 2 ;;
      *) args+=("$1"); shift ;;
    esac
  done
  podman build "${args[@]}" 1>&2
  dfile="$(mktemp)"
  podman push --tls-verify=false --digestfile "$dfile" "$tag" 1>&2
  printf '{"containerimage.digest":"%s"}' "$(cat "$dfile")" >"$meta"
  rm -f "$dfile"
  exit 0
fi
exec podman "$@"
EOF
  chmod +x "$SHIM/docker"
  export PATH="$SHIM:$PATH"
  export KIND_EXPERIMENTAL_PROVIDER=podman
  export KIND_CLUSTER_NAME
  export ATE_INSTALL_KIND=true
}

#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAGE_FILE="$(mktemp)"
WAS_RUNNING=0
export WEBOS_PORT="${WEBOS_PORT:-43173}"

if [[ -s "${APP_DIR}/.webos.pid" ]]; then
  read -r SERVER_PID < "${APP_DIR}/.webos.pid"
  if [[ "${SERVER_PID}" =~ ^[0-9]+$ ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    WAS_RUNNING=1
  fi
fi

cleanup() {
  unlink "${PAGE_FILE}" 2>/dev/null || true
  if [[ "${WAS_RUNNING}" -eq 0 ]]; then
    "${APP_DIR}/stop.sh"
  fi
}

trap cleanup EXIT

node --check "${APP_DIR}/app.js"
"${APP_DIR}/start.sh"
read -r ACTIVE_PORT < "${APP_DIR}/.webos.port"
curl -fsS "http://127.0.0.1:${ACTIVE_PORT}/" > "${PAGE_FILE}"
grep -q "LumaOS 2003" "${PAGE_FILE}"
grep -q "start-button" "${PAGE_FILE}"
test -s "${APP_DIR}/assets/wallpaper-collection.png"
echo "LumaOS checks passed"

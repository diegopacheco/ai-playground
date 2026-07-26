#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${APP_DIR}/.webos.pid"
PORT_FILE="${APP_DIR}/.webos.port"

if [[ ! -s "${PID_FILE}" || ! -s "${PORT_FILE}" ]]; then
  echo "LumaOS is not running"
  exit 0
fi

read -r SERVER_PID < "${PID_FILE}"
read -r ACTIVE_PORT < "${PORT_FILE}"
echo "LumaOS port: ${ACTIVE_PORT}"

if [[ ! "${SERVER_PID}" =~ ^[0-9]+$ ]] || [[ ! "${ACTIVE_PORT}" =~ ^[0-9]+$ ]]; then
  : > "${PID_FILE}"
  : > "${PORT_FILE}"
  echo "LumaOS has no valid server process"
  exit 0
fi

if kill -0 "${SERVER_PID}" 2>/dev/null; then
  kill "${SERVER_PID}"
  for _ in {1..30}; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      : > "${PID_FILE}"
      : > "${PORT_FILE}"
      echo "LumaOS stopped on port ${ACTIVE_PORT}"
      exit 0
    fi
    sleep 1
  done
  echo "LumaOS is still stopping on port ${ACTIVE_PORT}"
  exit 1
fi

: > "${PID_FILE}"
: > "${PORT_FILE}"
echo "LumaOS is not running on port ${ACTIVE_PORT}"

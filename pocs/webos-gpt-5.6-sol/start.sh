#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${APP_DIR}/.webos.pid"
PORT_FILE="${APP_DIR}/.webos.port"
LOG_FILE="${APP_DIR}/.webos.log"
START_PORT="${WEBOS_PORT:-4173}"

if [[ ! "${START_PORT}" =~ ^[0-9]+$ ]] || (( START_PORT < 1024 || START_PORT > 65535 )); then
  echo "LumaOS needs a starting port between 1024 and 65535"
  exit 1
fi

if [[ -s "${PID_FILE}" && -s "${PORT_FILE}" ]]; then
  read -r SERVER_PID < "${PID_FILE}"
  read -r ACTIVE_PORT < "${PORT_FILE}"
  if [[ "${SERVER_PID}" =~ ^[0-9]+$ ]] && [[ "${ACTIVE_PORT}" =~ ^[0-9]+$ ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    if curl -fsS "http://127.0.0.1:${ACTIVE_PORT}/api/health" 2>/dev/null | grep -q '"reader":true'; then
      echo "LumaOS port: ${ACTIVE_PORT}"
      echo "LumaOS is already running at http://127.0.0.1:${ACTIVE_PORT}"
      exit 0
    fi
    kill "${SERVER_PID}" 2>/dev/null || true
    for _ in {1..30}; do
      if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        break
      fi
      sleep 0.1
    done
  fi
fi

END_PORT=$((START_PORT + 99))
if (( END_PORT > 65535 )); then
  END_PORT=65535
fi

echo "LumaOS is searching for a port from ${START_PORT} to ${END_PORT}"

for (( PORT=START_PORT; PORT<=END_PORT; PORT++ )); do
  : > "${LOG_FILE}"
  python3 "${APP_DIR}/server.py" "${PORT}" "${APP_DIR}" > "${LOG_FILE}" 2>&1 &
  SERVER_PID=$!
  READY=0

  for _ in {1..20}; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/api/health" 2>/dev/null | grep -q '"reader":true'; then
      sleep 0.1
      if kill -0 "${SERVER_PID}" 2>/dev/null; then
        READY=1
      fi
      break
    fi
    sleep 0.1
  done

  if (( READY == 1 )); then
    echo "${SERVER_PID}" > "${PID_FILE}"
    echo "${PORT}" > "${PORT_FILE}"
    echo "LumaOS port: ${PORT}"
    echo "LumaOS is running at http://127.0.0.1:${PORT}"
    exit 0
  fi

  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
done

: > "${PID_FILE}"
: > "${PORT_FILE}"
echo "LumaOS could not find an available port from ${START_PORT} to ${END_PORT}"
echo "Read ${LOG_FILE}"
exit 1

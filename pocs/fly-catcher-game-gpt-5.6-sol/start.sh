#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .fly-catcher.state ]; then
  saved_pid="$(node -p "try{JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).pid}catch{''}")"
  if [ -n "$saved_pid" ] && kill -0 "$saved_pid" 2>/dev/null; then
    saved_http_port="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).httpPort")"
    saved_udp_port="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).udpPort")"
    saved_pairing_url="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).pairingUrl||''")"
    echo "Fly Catcher is already running at http://127.0.0.1:$saved_http_port"
    echo "iPhone controller UDP port: $saved_udp_port"
    if [ -n "$saved_pairing_url" ]; then
      echo "Phone pairing page: $saved_pairing_url"
    fi
    exit 0
  fi
fi
rm -f .fly-catcher.state .fly-catcher.pid
node server.js > fly-catcher.log 2>&1 &
pid=$!
for _ in $(seq 1 30); do
  if [ -f .fly-catcher.state ]; then
    saved_pid="$(node -p "try{JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).pid}catch{''}")"
    if [ "$saved_pid" = "$pid" ]; then
      saved_http_port="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).httpPort")"
      saved_udp_port="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).udpPort")"
      saved_pairing_url="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).pairingUrl||''")"
      if curl -fsS "http://127.0.0.1:$saved_http_port/api/status" >/dev/null 2>&1; then
        echo "Fly Catcher started at http://127.0.0.1:$saved_http_port"
        echo "iPhone controller sends UDP to this Mac on port $saved_udp_port"
        if [ -n "$saved_pairing_url" ]; then
          echo "Phone pairing page: $saved_pairing_url"
        fi
        exit 0
      fi
    fi
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    sed -n '1,120p' fly-catcher.log
    exit 1
  fi
  sleep 1
done
echo "Fly Catcher did not become ready"
exit 1

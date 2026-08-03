#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
was_running=false
if [ -f .fly-catcher.state ] && kill -0 "$(node -p "try{JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).pid}catch{''}")" 2>/dev/null; then
  was_running=true
else
  ./start.sh
fi
http_port="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).httpPort")"
udp_port="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).udpPort")"
finish() {
  if [ "$was_running" = false ]; then
    ./stop.sh
  fi
}
trap finish EXIT
page="$(curl -fsS "http://127.0.0.1:$http_port/")"
printf '%s' "$page" | grep -q "Fly Catcher"
GAME_PORT="$http_port" UDP_PORT="$udp_port" node tests/bridge.test.js
status="$(curl -fsS "http://127.0.0.1:$http_port/api/status")"
node -e "const s=JSON.parse(process.argv[1]); const p=Number(process.argv[2]); if(!s.running||s.packetCount<1||s.http!=='127.0.0.1:'+p) process.exit(1)" "$status" "$http_port"
pairing_url="$(node -p "JSON.parse(process.argv[1]).pairingUrl||''" "$status")"
if [ -n "$pairing_url" ]; then
  pairing_page="$(curl --noproxy '*' -fsS "$pairing_url")"
  printf '%s' "$pairing_page" | grep -q "Controller 1.1 required"
fi
node tests/ports.test.js
node tests/qr.test.js
node tests/layout.test.js
scheme="$(plutil -extract CFBundleURLTypes.0.CFBundleURLSchemes.0 raw ios/FlyCatcherController/Info.plist)"
if [ "$scheme" != "flycatcher" ]; then
  exit 1
fi
echo "HTTP game page passed"
echo "Private UDP motion packet passed"
echo "Browser event stream passed"
echo "Private pairing page passed"
echo "iPhone pairing address registration passed"
echo "Local-only game HTTP binding passed"

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
  printf '%s' "$pairing_page" | grep -q "Browser controller"
  printf '%s' "$pairing_page" | grep -q "Checking local trust"
  printf '%s' "$pairing_page" | grep -q "location.replace(target)"
  certificate_profile="$(curl --noproxy '*' -fsS "${pairing_url%/pair}/fly-catcher-ca.mobileconfig")"
  printf '%s' "$certificate_profile" | grep -q "Fly Catcher Local Root"
  controller_url="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).controllerUrl")"
  controller_token="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).controllerToken")"
  controller_page="$(curl --noproxy '*' --cacert .certs/ca.crt -fsS "$controller_url?token=$controller_token")"
  printf '%s' "$controller_page" | grep -q "Enable motion"
  health_url="${controller_url%/controller}/health.svg?token=$controller_token"
  health_status="$(curl --noproxy '*' --cacert .certs/ca.crt -sS -o /dev/null -w '%{http_code}' "$health_url")"
  if [ "$health_status" != "200" ]; then
    exit 1
  fi
  controller_api="${controller_url%/controller}/api/control"
  packets_before="$(node -p "JSON.parse(process.argv[1]).packetCount" "$status")"
  rejected_status="$(curl --noproxy '*' --cacert .certs/ca.crt -sS -o /dev/null -w '%{http_code}' -H "Content-Type: application/json" --data '{"type":"snap"}' "$controller_api")"
  if [ "$rejected_status" != "403" ]; then
    exit 1
  fi
  control_status="$(curl --noproxy '*' --cacert .certs/ca.crt -sS -o /dev/null -w '%{http_code}' -H "Content-Type: application/json" -H "X-Fly-Token: $controller_token" --data '{"type":"motion","ax":0.2,"ay":-0.1,"az":0.9}' "$controller_api")"
  if [ "$control_status" != "204" ]; then
    exit 1
  fi
  status_after="$(curl -fsS "http://127.0.0.1:$http_port/api/status")"
  node -e "const s=JSON.parse(process.argv[1]); if(s.packetCount<=Number(process.argv[2])) process.exit(1)" "$status_after" "$packets_before"
fi
node tests/ports.test.js
node tests/qr.test.js
node tests/layout.test.js
openssl verify -CAfile .certs/ca.crt .certs/server.crt >/dev/null
controller_host="$(node -p "JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).controllerHost")"
openssl x509 -in .certs/server.crt -noout -text | rg -q "IP Address:$controller_host"
echo "HTTP game page passed"
echo "Private UDP motion packet passed"
echo "Browser event stream passed"
echo "Private pairing page passed"
echo "Automatic trusted-controller redirect passed"
echo "Local certificate profile passed"
echo "HTTPS certificate chain and private IP passed"
echo "Invalid controller token rejection passed"
echo "HTTPS browser motion controller passed"
echo "Browser to UDP relay passed"
echo "Local-only game HTTP binding passed"

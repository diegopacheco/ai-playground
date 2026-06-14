#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PROM="http://localhost:9090"

ready=0
for i in $(seq 1 60); do
  if curl -sf "$PROM/-/ready" >/dev/null 2>&1; then ready=1; break; fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "FAIL: Prometheus not ready at $PROM"
  exit 1
fi
echo "Prometheus ready."

firing=0
for i in $(seq 1 60); do
  firing=$(curl -s "$PROM/api/v1/alerts" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(len([a for a in d["data"]["alerts"] if a["state"]=="firing"]))')
  if [ "$firing" -gt 0 ]; then break; fi
  sleep 1
done
if [ "$firing" -lt 1 ]; then
  echo "FAIL: no firing alerts after wait"
  exit 1
fi
echo "Firing alerts: $firing"
echo "---"
curl -s "$PROM/api/v1/alerts" | python3 -c 'import sys,json
d=json.load(sys.stdin)
for a in d["data"]["alerts"]:
    l=a["labels"]
    print(a["state"].upper(), l.get("alertname"), "severity="+l.get("severity","none"), "instance="+l.get("instance","?"))'
echo "---"
echo "PASS"

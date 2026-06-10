#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-7820}"
FAIL=0

node scan.js
if [ ! -f web/data.json ]; then echo "FAIL: web/data.json not generated"; exit 1; fi

node -e "const d=require('./web/data.json');if(!d.meta||typeof d.meta.totalPrompts!=='number'){console.error('FAIL: bad meta');process.exit(1)};if(!Array.isArray(d.tools)||!d.tools.length){console.error('FAIL: no tools');process.exit(1)};console.log('OK data.json: '+d.meta.totalPrompts+' prompts, '+d.tools.length+' tools')"

PORT="$PORT" node server.js > server.test.log 2>&1 &
SRV=$!
trap "kill $SRV 2>/dev/null" EXIT

UP=0
for i in $(seq 1 30); do
  if curl -s "http://127.0.0.1:${PORT}/data.json" -o /dev/null 2>/dev/null; then UP=1; break; fi
  sleep 1
done
if [ "$UP" != "1" ]; then echo "FAIL: server did not respond"; exit 1; fi

CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/")
if [ "$CODE" != "200" ]; then echo "FAIL: index returned $CODE"; FAIL=1; fi
JCODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/data.json")
if [ "$JCODE" != "200" ]; then echo "FAIL: data.json returned $JCODE"; FAIL=1; fi

if [ "$FAIL" = "0" ]; then
  echo "PASS: index ($CODE) and data.json ($JCODE) served on http://127.0.0.1:${PORT}"
else
  exit 1
fi

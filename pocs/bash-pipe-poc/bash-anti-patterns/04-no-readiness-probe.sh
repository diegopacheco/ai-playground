#!/usr/bin/env bash
# Anti-pattern: assuming 'started' equals 'ready'
# Source: design-doc.md §2.3, countered by G4
# Why it breaks: spawning a server and immediately calling curl returns
# Connection refused. The script reports failure even though the server
# was about to come up 1s later.

set -euo pipefail

echo "ANTI-PATTERN: no readiness probe -- treating 'started' as 'ready'"
echo "WHY: a server takes time to bind its port; an immediate curl will fail."
echo
echo "DEMO: spawn a fake server that opens its port after a small delay,"
echo "then curl it immediately with no probe."

port=$(( ( RANDOM % 1000 ) + 18000 ))
(
  sleep 1
  if command -v nc >/dev/null 2>&1; then
    printf 'HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\nOK' | nc -l "$port" >/dev/null 2>&1 &
  fi
) &
spawner=$!

if curl -fsS --max-time 1 "http://127.0.0.1:$port/" >/dev/null 2>&1; then
  echo "RESULT: curl succeeded (unexpected on a cold spawn)"
else
  echo "RESULT: curl failed immediately -- 'Connection refused' / timeout."
  echo "  Real bug: the server WAS coming up; we just did not wait."
fi

wait "$spawner" 2>/dev/null || true
pkill -P $$ 2>/dev/null || true
echo
echo "Correct form (G4): bounded poll loop with curl --max-time 2, up to 60 tries."

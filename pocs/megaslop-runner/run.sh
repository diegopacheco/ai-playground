#!/bin/bash
DIR=$(cd "$(dirname "$0")" && pwd)
PORT=8088
lsof -ti:$PORT | xargs kill 2>/dev/null
(cd "$DIR" && python3 -m http.server $PORT) &
PID=$!
echo "$PID" > /tmp/megaslop-runner.pid
URL="http://localhost:$PORT"
echo "MEGASLOP RUNNER serving at $URL"
for i in $(seq 1 20); do
  if curl -s -o /dev/null "$URL"; then
    break
  fi
  sleep 1
done
if command -v open >/dev/null 2>&1; then
  open "$URL"
fi
echo "PID $PID written to /tmp/megaslop-runner.pid"
echo "Run ./stop.sh to stop."
wait $PID

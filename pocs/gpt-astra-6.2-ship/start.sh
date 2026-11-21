#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
url=http://127.0.0.1:8347/
ready() {
  curl --noproxy '*' -fsS "$url" 2>/dev/null | cmp -s "$root/index.html" -
}
if ready; then
  echo "$url"
  exit 0
fi
mkdir -p "$root/.run"
nohup python3 -m http.server 8347 --bind 127.0.0.1 --directory "$root" >"$root/.run/server.log" 2>&1 </dev/null &
pid=$!
for ((i=0;i<10;i++)); do
  if ! kill -0 "$pid" 2>/dev/null; then
    cat "$root/.run/server.log" >&2
    exit 1
  fi
  if ready; then
    echo "$url"
    exit 0
  fi
  sleep 1
done
kill "$pid"
echo "Server failed to serve index.html" >&2
exit 1

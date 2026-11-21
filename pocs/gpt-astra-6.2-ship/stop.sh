#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid="$(lsof -nP -t -iTCP@127.0.0.1:8347 -sTCP:LISTEN || true)"
if [[ -z "$pid" ]]; then
  exit 0
fi
command="$(ps -p "$pid" -o command=)"
case "$command" in
  *" -m http.server 8347 --bind 127.0.0.1 --directory $root") ;;
  *" -m http.server 8347 --bind 127.0.0.1")
    lsof -a -p "$pid" -d cwd -Fn | grep -Fxq "n$root" || exit 1 ;;
  *) echo "Port 8347 belongs to another process" >&2; exit 1 ;;
esac
kill "$pid"
for ((i=0;i<10;i++)); do
  if ! lsof -a -p "$pid" -iTCP:8347 -sTCP:LISTEN >/dev/null 2>&1; then
    exit 0
  fi
  sleep 1
done
exit 1

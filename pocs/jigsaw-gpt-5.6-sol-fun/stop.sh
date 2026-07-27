#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -f .jigsaw.pid ]; then
  exit 0
fi
process_id=$(cat .jigsaw.pid)
if kill -0 "$process_id" 2>/dev/null; then
  kill "$process_id"
fi
rm -f .jigsaw.pid
echo "Big Cat Jigsaw stopped"

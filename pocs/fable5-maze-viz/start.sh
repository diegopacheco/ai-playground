#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p out
javac -d out src/*.java
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "already running at http://localhost:8013"
  exit 0
fi
nohup java -cp out Server 8013 > server.log 2>&1 &
echo $! > server.pid
for i in $(seq 1 30); do
  if curl -s -o /dev/null http://localhost:8013/; then
    echo "maze race running at http://localhost:8013"
    exit 0
  fi
  sleep 1
done
echo "server failed to start, check server.log"
exit 1

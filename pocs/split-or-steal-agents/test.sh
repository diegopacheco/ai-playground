#!/bin/bash
cd "$(dirname "$0")"
lsof -ti tcp:8017 | xargs kill -9 2>/dev/null
SPLIT_OR_STEAL_FAKE=1 python3 server.py >/dev/null 2>&1 &
pid=$!
for i in $(seq 1 20); do
  curl -s -o /dev/null http://localhost:8017/ && break
  sleep 1
done
pass=0
fail=0
check() {
  if [ "$1" = "0" ]; then
    echo "PASS: $2"
    pass=$((pass+1))
  else
    echo "FAIL: $2"
    fail=$((fail+1))
  fi
}
curl -s http://localhost:8017/ | grep -q "Split or Steal"
check $? "page served"
curl -s http://localhost:8017/state | grep -q '"status": "idle"'
check $? "state endpoint idle"
curl -s -X POST "http://localhost:8017/start?rounds=2" | grep -q '"ok":true'
check $? "game start accepted"
done=1
for i in $(seq 1 30); do
  if curl -s http://localhost:8017/state | grep -q '"status": "done"'; then
    done=0
    break
  fi
  sleep 1
done
check $done "2 fake rounds completed"
state=$(curl -s http://localhost:8017/state)
echo "$state" | grep -q '"type": "msg"'
check $? "negotiation messages logged"
echo "$state" | grep -q '"type": "reveal"'
check $? "decisions revealed"
echo "$state" | grep -q '"verdict"'
check $? "verdict announced"
kill $pid 2>/dev/null
echo "passed $pass, failed $fail"
[ "$fail" -eq 0 ]

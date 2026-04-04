#!/bin/bash
PASS=0
FAIL=0

check() {
  local desc="$1"
  local expected="$2"
  local actual="$3"
  if echo "$actual" | grep -q "$expected"; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc (expected '$expected', got '$actual')"
    FAIL=$((FAIL + 1))
  fi
}

cargo build --release
BIN=./target/release/redis-fs

OUTPUT=$(echo -e "create /hello.txt Hello World\nexit" | $BIN)
check "create file" "created /hello.txt" "$OUTPUT"

OUTPUT=$(echo -e "cat /hello.txt\nexit" | $BIN)
check "read file" "Hello World" "$OUTPUT"

OUTPUT=$(echo -e "cp /hello.txt /copy.txt\nexit" | $BIN)
check "copy file" "copied /hello.txt -> /copy.txt" "$OUTPUT"

OUTPUT=$(echo -e "cat /copy.txt\nexit" | $BIN)
check "read copy" "Hello World" "$OUTPUT"

OUTPUT=$(echo -e "ls /\nexit" | $BIN)
check "list has hello.txt" "hello.txt" "$OUTPUT"
check "list has copy.txt" "copy.txt" "$OUTPUT"

OUTPUT=$(echo -e "mkdir /scripts\nexit" | $BIN)
check "mkdir" "created directory /scripts" "$OUTPUT"

OUTPUT=$(echo -e "ls /\nexit" | $BIN)
check "list has scripts" "scripts" "$OUTPUT"

OUTPUT=$(echo -e "create /scripts/run.sh echo redis-fs-works\nexit" | $BIN)
check "create script" "created /scripts/run.sh" "$OUTPUT"

OUTPUT=$(echo -e "exec /scripts/run.sh\nexit" | $BIN)
check "exec script" "redis-fs-works" "$OUTPUT"

OUTPUT=$(echo -e "rm /hello.txt\nexit" | $BIN)
check "delete file" "removed /hello.txt" "$OUTPUT"

OUTPUT=$(echo -e "cat /hello.txt\nexit" | $BIN)
check "verify deletion" "not found" "$OUTPUT"

OUTPUT=$(echo -e "rm /copy.txt\nexit" | $BIN)
check "delete copy" "removed /copy.txt" "$OUTPUT"

OUTPUT=$(echo -e "ls /\nexit" | $BIN)
check "list after deletes" "scripts" "$OUTPUT"

OUTPUT=$(echo -e "cd /scripts\npwd\nexit" | $BIN)
check "cd and pwd" "/scripts" "$OUTPUT"

echo ""
echo "Results: $PASS passed, $FAIL failed"

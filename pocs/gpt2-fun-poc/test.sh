#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/vendor/gpt2.cmake"
PASS=0
FAIL=0

check() {
  local name="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    echo "PASS  $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL  $name"
    echo "      expected: $expected"
    echo "      actual:   $actual"
    FAIL=$((FAIL + 1))
  fi
}

if [ ! -f "$VENDOR/data/expected.txt" ]; then
  echo "weights not generated, run ./build.sh first"
  exit 1
fi

echo "=== CMake integer semantics the Q16.16 kernels rely on ==="
PROBE="$(cmake -P "$VENDOR/probe.cmake" 2>&1)"
check "64-bit signed integers" \
  "9223372036854775807" "$(echo "$PROBE" | sed -n 's/^max_int(2\^63-1): //p')"
check "right shift is arithmetic, not logical" \
  "-4" "$(echo "$PROBE" | sed -n 's/^-8 >> 1: //p')"
check "division truncates toward zero" \
  "-3" "$(echo "$PROBE" | sed -n 's|^-7 / 2: ||p')"
check "bitwise and is supported" \
  "1" "$(echo "$PROBE" | sed -n 's/^3 & 5: //p')"

echo
echo "=== forward pass matches the Python reference bit-for-bit ==="
GOLDEN_IDS="$(sed -n 's/^ids: //p' "$VENDOR/data/expected.txt" | tr -d '[] ' | tr ',' ';')"
GOLDEN_TEXT="$(sed -n 's/^text: //p' "$VENDOR/data/expected.txt")"
GOLDEN_PROMPT="$(sed -n 's/^prompt: //p' "$VENDOR/data/expected.txt")"
OUT="$(cmake -DPROMPT="$GOLDEN_PROMPT" -DN=6 -P "$VENDOR/gpt2.cmake" 2>&1)"
check "token ids match golden" "$GOLDEN_IDS" "$(echo "$OUT" | sed -n 's/^ids:  //p')"
check "decoded text matches golden" "$GOLDEN_TEXT" "$(echo "$OUT" | sed -n 's/^text: //p')"

echo
echo "=== greedy decoding is deterministic ==="
A="$(cmake -DPROMPT="a cat" -DN=4 -P "$VENDOR/gpt2.cmake" 2>&1 | sed -n 's/^ids:  //p')"
B="$(cmake -DPROMPT="a cat" -DN=4 -P "$VENDOR/gpt2.cmake" 2>&1 | sed -n 's/^ids:  //p')"
check "two runs of the same prompt agree" "$A" "$B"

echo
echo "=== generation length is prompt plus N ==="
IDS="$(cmake -DPROMPT="the dog" -DN=5 -P "$VENDOR/gpt2.cmake" 2>&1 | sed -n 's/^ids:  //p')"
check "7 prompt chars plus 5 generated" "12" "$(echo "$IDS" | tr ';' '\n' | grep -c .)"

echo
echo "=== run.sh rejects input the toy model cannot represent ==="
OUT="$(./run.sh "Hello" 4 2>&1)"
check "characters outside the 32-char vocab are refused" \
  "1" "$(echo "$OUT" | grep -c 'outside the toy vocab')"
OUT="$(./run.sh "hello world" 8 2>&1)"
check "overflowing the context window is refused" \
  "1" "$(echo "$OUT" | grep -c 'exceeds the 16-token context window')"

echo
echo "passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ]

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/vendor/gpt2.cmake"
PROMPT="${1:-hi}"
N="${2:-6}"

if [ ! -f "$VENDOR/data/model.cmake" ]; then
  echo "weights not generated, run ./build.sh first"
  exit 1
fi

N_CTX="$(sed -n 's/^set(N_CTX \([0-9]*\))$/\1/p' "$VENDOR/data/model.cmake")"
VOCAB="$(sed -n 's/^set(VOCAB "\(.*\)")$/\1/p' "$VENDOR/data/model.cmake")"
LEN="${#PROMPT}"

if [ "$((LEN + N))" -gt "$N_CTX" ]; then
  echo "prompt ($LEN chars) plus $N generated tokens exceeds the $N_CTX-token context window"
  exit 1
fi

case "$PROMPT" in
  *[!\ abcdefghijklmnopqrstuvwxyz.,\'!?]*)
    echo "prompt has characters outside the toy vocab: [$VOCAB]"
    exit 1
    ;;
esac

echo "prompt   ${PROMPT}"
echo "tokens   ${N}"
echo "context  ${LEN}/${N_CTX} used by the prompt"
echo "engine   cmake -P, Q16.16 integer math, no compiler involved"
echo

cmake -DPROMPT="$PROMPT" -DN="$N" -P "$VENDOR/gpt2.cmake"

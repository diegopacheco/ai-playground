#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/vendor/gpt2.cmake"
CKPT="$VENDOR/checkpoint"
DATA="$VENDOR/data/gpt2_full.cmake"
BASE="https://huggingface.co/openai-community/gpt2/resolve/main"
PROMPT="${1:-Hello}"
N="${2:-1}"

if [ ! -d "$VENDOR/.git" ]; then
  echo "upstream not cloned, run ./build.sh first"
  exit 1
fi

mkdir -p "$CKPT"
for f in model.safetensors vocab.json merges.txt; do
  if [ ! -s "$CKPT/$f" ]; then
    echo "downloading $f"
    curl -fsSL -o "$CKPT/$f" "$BASE/$f"
  fi
done

if [ ! -s "$DATA" ]; then
  echo "converting the checkpoint to Q16.16 CMake, this writes about 2.4 GB"
  python3 "$VENDOR/tools/gen_full.py" \
    "$CKPT/model.safetensors" "$CKPT/vocab.json" "$CKPT/merges.txt" "$PROMPT" 0
fi

echo
echo "prompt   ${PROMPT}"
echo "tokens   ${N}"
echo "weights  $(du -h "$DATA" | cut -f1) of CMake source, 124M parameters"
echo "engine   cmake -P, Q16.16 integer math, no compiler involved"
echo

cmake -DPROMPT="$PROMPT" -DN="$N" -P "$VENDOR/gpt2_full.cmake"

echo
echo "reclaim the disk with: rm -rf $VENDOR/checkpoint $VENDOR/data/gpt2_full.cmake"

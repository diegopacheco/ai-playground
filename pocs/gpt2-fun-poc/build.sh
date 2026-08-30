#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/vendor/gpt2.cmake"
REPO="https://github.com/AlpinDale/gpt2.cmake.git"
COMMIT="f04ef6abfe435eb3be46380aeb21f4947d443389"

for tool in cmake python3 git; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing required tool: $tool"
    exit 1
  fi
done

if [ ! -d "$VENDOR/.git" ]; then
  rm -rf "$VENDOR"
  mkdir -p "$ROOT/vendor"
  git clone "$REPO" "$VENDOR"
fi

git -C "$VENDOR" fetch origin
git -C "$VENDOR" checkout "$COMMIT"

python3 "$VENDOR/tools/gen_tables.py"
python3 "$VENDOR/tools/gen_model.py"

echo
echo "build ok"
echo "cmake    $(cmake --version | head -1)"
echo "python   $(python3 --version)"
echo "vendor   $VENDOR"
echo "commit   $COMMIT"
echo "tables   $VENDOR/cmake/tables.cmake"
echo "weights  $VENDOR/data/model.cmake"
echo "golden   $VENDOR/data/expected.txt"

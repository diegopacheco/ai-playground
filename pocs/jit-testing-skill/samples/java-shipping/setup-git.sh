#!/usr/bin/env bash
set -euo pipefail

SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SAMPLE_DIR}"

rm -rf .git

git init -q
git config user.email "sample@local"
git config user.name "sample"

tmp_diff="$(mktemp -d)"
cp -R src "${tmp_diff}/"

rm -rf src
cp -R .parent/src ./

git add -A
git commit -q -m "parent: shipping by total threshold"

rm -rf src
cp -R "${tmp_diff}/src" ./

rm -rf "${tmp_diff}"

git add -A
git commit -q -m "Add free shipping for store-pickup orders"

echo "ready. run /jit from inside this dir."

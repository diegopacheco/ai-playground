#!/usr/bin/env bash
set -euo pipefail

SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SAMPLE_DIR}"

rm -rf .git
git init -q
git config user.email "sample@local"
git config user.name "sample"

tmp_diff="$(mktemp -d)"
cp src/discount.js "${tmp_diff}/"

cp .parent/src/discount.js src/discount.js
git add package.json src/discount.js
git commit -q -m "parent: shipping by total threshold"

cp "${tmp_diff}/discount.js" src/discount.js
rm -rf "${tmp_diff}"
git add src/discount.js
git commit -q -m "Add free shipping for store-pickup orders"

echo "ready. run /jit from inside this dir."

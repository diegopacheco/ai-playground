#!/usr/bin/env bash
set -euo pipefail

SAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SAMPLE_DIR}"

rm -rf .git
git init -q
git config user.email "sample@local"
git config user.name "sample"

tmp_diff="$(mktemp -d)"
cp pricing.py "${tmp_diff}/"

cp .parent/pricing.py ./pricing.py
git add pricing.py pyproject.toml
git commit -q -m "parent: shipping by total threshold"

cp "${tmp_diff}/pricing.py" ./pricing.py
rm -rf "${tmp_diff}"
git add pricing.py
git commit -q -m "Add free shipping for store-pickup orders"

echo "ready. run /jit from inside this dir."

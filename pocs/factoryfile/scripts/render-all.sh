#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

cargo build -q --release --manifest-path "$ROOT/cli/Cargo.toml"
cli="$ROOT/cli/target/release/factory-cli"
mkdir -p "$ROOT/landing/factories"
for file in "$ROOT"/samples/*.factoryfile; do
  "$cli" --render-ui "$file" --out "$ROOT/landing/factories/$(basename "$file" .factoryfile).html" --no-open
done
"$cli" --render-ui "$ROOT/Factoryfile" --out "$ROOT/landing/factories/factoryfile.html" --no-open

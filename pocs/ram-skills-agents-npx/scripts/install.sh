#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

require cargo
cargo install --quiet --force --path "$ROOT"
echo "installed ram into $(dirname "$(command -v ram || echo "$HOME/.cargo/bin/ram")")"

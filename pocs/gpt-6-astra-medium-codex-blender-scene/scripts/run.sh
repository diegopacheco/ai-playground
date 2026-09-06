#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
"$ROOT/scripts/build.sh"
"$ROOT/scripts/render.sh"
"$ROOT/scripts/export.sh"

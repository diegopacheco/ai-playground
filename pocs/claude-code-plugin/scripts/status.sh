#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
require_claude

if marketplace_added; then
  echo "marketplace $MARKETPLACE ADDED"
else
  echo "marketplace $MARKETPLACE MISSING"
fi

for p in $PLUGINS; do
  if plugin_installed "$p"; then
    echo "$p@$MARKETPLACE INSTALLED"
  else
    echo "$p@$MARKETPLACE NOT INSTALLED"
  fi
done

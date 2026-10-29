#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
require_claude

selected="$(pick_plugins uninstall "$@")"
if [ -z "${selected// /}" ]; then
  echo "nothing selected"
  exit 0
fi

for p in $selected; do
  if plugin_installed "$p"; then
    claude plugin uninstall "$p@$MARKETPLACE" --scope "$SCOPE"
  else
    echo "$p@$MARKETPLACE not installed"
  fi
done

remaining=0
for p in $PLUGINS; do
  if plugin_installed "$p"; then
    remaining=1
  fi
done

if [ "$remaining" -eq 0 ] && marketplace_added; then
  claude plugin marketplace remove "$MARKETPLACE"
fi

echo "uninstalled:$selected"

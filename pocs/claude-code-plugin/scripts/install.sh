#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"
require_claude

selected="$(pick_plugins install "$@")"
if [ -z "${selected// /}" ]; then
  echo "nothing selected"
  exit 0
fi

if marketplace_added; then
  echo "marketplace $MARKETPLACE already added"
else
  claude plugin marketplace add "$ROOT" --scope "$SCOPE"
fi

for p in $selected; do
  if plugin_installed "$p"; then
    echo "$p@$MARKETPLACE already installed"
  else
    claude plugin install "$p@$MARKETPLACE" --scope "$SCOPE"
  fi
done

echo "installed:$selected"
echo "restart Claude Code or run /reload-plugins to load the skills"

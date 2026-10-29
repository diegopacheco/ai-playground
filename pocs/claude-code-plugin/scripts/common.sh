#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKETPLACE="dev-trio"
PLUGINS="code debug troubleshoot"
SCOPE="${SCOPE:-user}"

require_claude() {
  if ! command -v claude >/dev/null 2>&1; then
    echo "claude CLI not found in PATH" >&2
    exit 1
  fi
}

is_valid_plugin() {
  local name="$1"
  local p
  for p in $PLUGINS; do
    if [ "$p" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

marketplace_added() {
  local out
  out="$(claude plugin marketplace list --json)"
  echo "$out" | grep "\"name\": \"$MARKETPLACE\"" >/dev/null
}

plugin_installed() {
  local out
  out="$(claude plugin list --json)"
  echo "$out" | grep "\"id\": \"$1@$MARKETPLACE\"" >/dev/null
}

pick_plugins() {
  local action="$1"
  shift
  local selected=""
  local p answer
  if [ "$#" -eq 0 ]; then
    if [ ! -t 0 ]; then
      echo "no plugins given and no terminal to ask; pass names or all: $PLUGINS" >&2
      exit 1
    fi
    for p in $PLUGINS; do
      printf "%s %s? [y/N] " "$action" "$p" >&2
      read -r answer
      case "$answer" in
        y|Y|yes|YES) selected="$selected $p" ;;
      esac
    done
  else
    for p in "$@"; do
      if [ "$p" = "all" ]; then
        selected="$PLUGINS"
        break
      fi
      if ! is_valid_plugin "$p"; then
        echo "unknown plugin: $p (valid: $PLUGINS all)" >&2
        exit 1
      fi
      selected="$selected $p"
    done
  fi
  echo "$selected"
}

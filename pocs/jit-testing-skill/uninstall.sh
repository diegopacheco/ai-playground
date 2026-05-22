#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HOME}/.claude/jit"
SKILLS_DIR="${HOME}/.claude/skills"
GLOBAL_IGNORE="${HOME}/.config/git/ignore"

rm -rf "${INSTALL_DIR}"
rm -rf "${SKILLS_DIR}/jit"
rm -rf "${SKILLS_DIR}/jit-dashboard"

if [ -f "${GLOBAL_IGNORE}" ]; then
  tmp="$(mktemp)"
  grep -vxF ".jit-testing/" "${GLOBAL_IGNORE}" > "${tmp}" || true
  mv "${tmp}" "${GLOBAL_IGNORE}"
fi

echo "uninstalled jit"

if [ "${1:-}" = "--purge-runs" ]; then
  ROOT="${2:-${HOME}}"
  find "${ROOT}" -type d -name ".jit-testing" -prune -exec rm -rf {} +
  echo "purged .jit-testing dirs under ${ROOT}"
fi

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${HOME}/.claude/jit"
SKILLS_DIR="${HOME}/.claude/skills"

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
mkdir -p "${SKILLS_DIR}/jit"
mkdir -p "${SKILLS_DIR}/jit-dashboard"

cp -R "${SCRIPT_DIR}/bin" "${INSTALL_DIR}/"
cp -R "${SCRIPT_DIR}/lib" "${INSTALL_DIR}/"
cp -R "${SCRIPT_DIR}/dashboard" "${INSTALL_DIR}/"
cp -R "${SCRIPT_DIR}/samples" "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/bin/jit"

cp "${SCRIPT_DIR}/SKILL.md" "${SKILLS_DIR}/jit/SKILL.md"
cp "${SCRIPT_DIR}/commands/jit-dashboard.md" "${SKILLS_DIR}/jit-dashboard/SKILL.md"

GLOBAL_IGNORE_DIR="${HOME}/.config/git"
GLOBAL_IGNORE="${GLOBAL_IGNORE_DIR}/ignore"
mkdir -p "${GLOBAL_IGNORE_DIR}"
touch "${GLOBAL_IGNORE}"
if ! grep -qxF ".jit-testing/" "${GLOBAL_IGNORE}"; then
  echo ".jit-testing/" >> "${GLOBAL_IGNORE}"
fi

echo "installed to ${INSTALL_DIR}"
echo "skills installed: /jit /jit-dashboard"
echo ""
echo "target availability:"

check_cmd() {
  local label="$1"
  local cmd="$2"
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "  ok    ${label} (${cmd})"
  else
    echo "  miss  ${label} (${cmd} not found)"
  fi
}

check_cmd "java"      "java"
check_cmd "javac"     "javac"
check_cmd "maven"     "mvn"
check_cmd "gradle"    "gradle"
check_cmd "kotlin"    "kotlinc"
check_cmd "sbt"       "sbt"
check_cmd "bazel"     "bazel"
check_cmd "python3"   "python3"
check_cmd "node"      "node"

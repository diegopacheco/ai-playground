#!/usr/bin/env bash
set -euo pipefail

DEST="${HOME}/.claude/skills"
rm -rf "${DEST}/bug-recording" "${DEST}/bug-report"
echo "removed bug-recording and bug-report from ${DEST}"

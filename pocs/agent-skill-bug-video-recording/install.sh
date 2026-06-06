#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${HOME}/.claude/skills"

if ! command -v node >/dev/null 2>&1 && ! command -v bun >/dev/null 2>&1; then
  echo "node or bun is required"
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required"
  exit 1
fi

mkdir -p "${DEST}"
rm -rf "${DEST}/bug-recording" "${DEST}/bug-report"
cp -R "${SRC}/skills/bug-recording" "${DEST}/bug-recording"
cp -R "${SRC}/skills/bug-report" "${DEST}/bug-report"
rm -rf "${DEST}/bug-recording/node_modules"

cd "${DEST}/bug-recording"
npm install --silent
npx --yes playwright install chromium

chmod +x "${DEST}/bug-recording/record.mjs" "${DEST}/bug-recording/reduce-size.sh" "${DEST}/bug-report/report.mjs"

echo "installed bug-recording -> ${DEST}/bug-recording"
echo "installed bug-report -> ${DEST}/bug-report"

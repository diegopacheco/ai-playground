#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/skills/skill-sec-checker/scripts"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/skill-sec-checker-sample-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

export HOME="$WORK/home"
mkdir -p "$HOME/.claude/skills" "$HOME/.codex/skills"
cp -R "$ROOT/sample/skills/claude/." "$HOME/.claude/skills/"
cp -R "$ROOT/sample/skills/codex/." "$HOME/.codex/skills/"

run() {
  export SKILL_SEC_CHECKER_NOW="$1"
  local dir
  dir="$(python3 "$SCRIPTS/scan.py" | sed -n 's/^run dir: //p')"
  if [ -n "$2" ]; then cp "$ROOT/sample/reviews/$2" "$dir/review.json"; fi
  python3 "$SCRIPTS/render.py" "$dir" >/dev/null
  echo "$dir"
}

run 2026-08-30T09:30:00Z run-1.json >/dev/null
run 2026-09-06T09:30:00Z "" >/dev/null
cp -R "$ROOT/sample/changes/claude/." "$HOME/.claude/skills/"
cp -R "$ROOT/sample/changes/codex/." "$HOME/.codex/skills/"
rm -rf "$HOME/.claude/skills/weather-brief"
LAST="$(run 2026-09-13T09:30:00Z run-3.json)"

cp "$LAST/index.html" "$ROOT/sample/index.html"
cp "$HOME/.skill-sec-checker/history.json" "$ROOT/sample/history.json"
echo "sample report: $ROOT/sample/index.html"

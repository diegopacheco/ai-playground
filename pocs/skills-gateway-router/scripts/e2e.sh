#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TITLE="Olá Mundo: Skills Gateway Router!"
SLUG="ola-mundo-skills-gateway-router"
NOTES="one two three four five six seven"
WORDS="7"
PROMPT="Use the $SKILL_NAME skill for both tasks. First turn the title '$TITLE' into a URL slug. Then count the words in the file notes.txt. Reply with the slug and the word count."

run_claude() {
  claude -p "$PROMPT" --output-format stream-json --verbose --allowedTools "Skill,Bash(node:*),Read" --max-turns 25 </dev/null
}

run_codex() {
  codex exec --json --skip-git-repo-check --sandbox read-only "$PROMPT" </dev/null
}

check() {
  local agent transcript
  agent="$1"
  transcript="$2"
  grep -qF "$SLUG" "$transcript" || fail "$agent: answer has no slug $SLUG, see $transcript"
  grep -qE 'node [^ ]+ route' "$transcript" || fail "$agent: never ran the gateway route command, see $transcript"
  grep -qF "skill: text__slugify" "$transcript" || fail "$agent: never loaded text__slugify through the gateway, see $transcript"
  grep -qF "skill: text__word-count" "$transcript" || fail "$agent: never loaded text__word-count through the gateway, see $transcript"
  grep -qE "(^|[^0-9])$WORDS([^0-9]|$)" "$transcript" || fail "$agent: answer has no word count $WORDS, see $transcript"
}

agents="$(agents_for "${1:-all}")"
for agent in $agents; do
  require "$agent"
  is_our_install "$(agent_skills_dir "$agent")/$SKILL_NAME" || fail "$agent: $SKILL_NAME is not installed, run ./scripts/install.sh $agent"
  workdir="$(mktemp -d)"
  printf "%s\n" "$NOTES" >"$workdir/notes.txt"
  transcript="$LOGS/e2e-$agent.jsonl"
  log "$agent: running in $workdir, transcript $transcript"
  (cd "$workdir" && "run_$agent") >"$transcript" 2>&1 || fail "$agent: CLI failed, see $transcript"
  check "$agent" "$transcript"
  rm -rf "$workdir"
  log "$agent: PASS routed through $SKILL_NAME to text__slugify and text__word-count"
done

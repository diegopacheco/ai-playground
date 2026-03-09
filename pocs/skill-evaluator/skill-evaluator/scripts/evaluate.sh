#!/bin/bash
SKILL_PATH="$1"

if [ -z "$SKILL_PATH" ]; then
  echo "Usage: evaluate.sh <path-to-skill-directory>"
  exit 1
fi

if [ ! -d "$SKILL_PATH" ]; then
  echo "ERROR: Directory not found: $SKILL_PATH"
  exit 1
fi

SKILL_MD="$SKILL_PATH/SKILL.md"
SKILL_NAME=$(basename "$SKILL_PATH")

echo "=== SKILL EVALUATION: $SKILL_NAME ==="
echo "Path: $SKILL_PATH"
echo ""

if [ ! -f "$SKILL_MD" ]; then
  echo "STRUCTURE: SKILL.md NOT FOUND"
  exit 1
fi
echo "STRUCTURE: SKILL.md found"

FRONTMATTER=$(sed -n '/^---$/,/^---$/p' "$SKILL_MD")
if [ -z "$FRONTMATTER" ]; then
  echo "FRONTMATTER: missing"
else
  echo "FRONTMATTER: present"
  HAS_NAME=$(echo "$FRONTMATTER" | grep -c "^name:")
  HAS_DESC=$(echo "$FRONTMATTER" | grep -c "^description:")
  echo "FRONTMATTER_NAME: $HAS_NAME"
  echo "FRONTMATTER_DESC: $HAS_DESC"
fi

BODY=$(sed '1,/^---$/d' "$SKILL_MD" | sed '1,/^---$/d')
WORD_COUNT=$(echo "$BODY" | wc -w | tr -d ' ')
LINE_COUNT=$(echo "$BODY" | wc -l | tr -d ' ')
echo "BODY_WORDS: $WORD_COUNT"
echo "BODY_LINES: $LINE_COUNT"

if [ "$WORD_COUNT" -le 2000 ]; then
  echo "TOKEN_EFFICIENCY: good (under 2000 words)"
elif [ "$WORD_COUNT" -le 3000 ]; then
  echo "TOKEN_EFFICIENCY: acceptable (under 3000 words)"
else
  echo "TOKEN_EFFICIENCY: bloated ($WORD_COUNT words)"
fi

FILLER_COUNT=$(echo "$BODY" | grep -ioE '\b(please|kindly|basically|essentially|simply|just|maybe|perhaps|might|could potentially)\b' | wc -l | tr -d ' ')
echo "FILLER_WORDS: $FILLER_COUNT"

VAGUE_VERBS=$(echo "$BODY" | grep -ioE '\b(handle|manage|process|deal with|take care of)\b' | wc -l | tr -d ' ')
echo "VAGUE_VERBS: $VAGUE_VERBS"

ANTICHEAT=$(echo "$BODY" | grep -icE '(cheat|fake|fabricat|hardcod|hallucin|never skip|must verify|must validate|correctness)' | tr -d ' ')
echo "ANTICHEAT_SIGNALS: $ANTICHEAT"

GATES=$(echo "$BODY" | grep -icE '(must pass|do not continue|do not proceed|cannot skip|before proceeding|gate|checkpoint|block)' | tr -d ' ')
echo "QUALITY_GATES: $GATES"

PINNED_VERSIONS=$(echo "$BODY" | grep -cE '[0-9]+\.[0-9]+' | tr -d ' ')
echo "PINNED_VERSIONS: $PINNED_VERSIONS"

SCOPE_SIGNALS=$(echo "$BODY" | grep -icE '(only create|do not modify|do not add|do not refactor|minimal|no extra|stay focused|in scope)' | tr -d ' ')
echo "SCOPE_DISCIPLINE_SIGNALS: $SCOPE_SIGNALS"

ERROR_SIGNALS=$(echo "$BODY" | grep -icE '(rollback|retry|revert|on failure|if.*fails|error handling|graceful|recover)' | tr -d ' ')
echo "ERROR_RECOVERY_SIGNALS: $ERROR_SIGNALS"

OBSERVE_SIGNALS=$(echo "$BODY" | grep -icE '(log|report|output|diff|evidence|audit|trail|findings|before.*after|progress)' | tr -d ' ')
echo "OBSERVABILITY_SIGNALS: $OBSERVE_SIGNALS"

IDEMPOTENT_SIGNALS=$(echo "$BODY" | grep -icE '(idempoten|re-run|rerun|already exists|skip if|detect prior|safe to run)' | tr -d ' ')
echo "IDEMPOTENCY_SIGNALS: $IDEMPOTENT_SIGNALS"

echo ""
echo "=== FILES ==="
find "$SKILL_PATH" -type f | while read -r f; do
  REL=$(echo "$f" | sed "s|$SKILL_PATH/||")
  WC=$(wc -w < "$f" | tr -d ' ')
  echo "FILE: $REL ($WC words)"
done

HAS_REFS=0
HAS_SCRIPTS=0
HAS_ASSETS=0
[ -d "$SKILL_PATH/references" ] && HAS_REFS=1
[ -d "$SKILL_PATH/scripts" ] && HAS_SCRIPTS=1
[ -d "$SKILL_PATH/assets" ] && HAS_ASSETS=1
echo ""
echo "HAS_REFERENCES_DIR: $HAS_REFS"
echo "HAS_SCRIPTS_DIR: $HAS_SCRIPTS"
echo "HAS_ASSETS_DIR: $HAS_ASSETS"

REFERENCED_FILES=$(echo "$BODY" | grep -oE '`(references|scripts|assets)/[^`]+`' | tr -d '`' | sort -u)
echo ""
echo "=== REFERENCED FILES CHECK ==="
if [ -z "$REFERENCED_FILES" ]; then
  echo "NO_FILE_REFERENCES: true"
else
  echo "$REFERENCED_FILES" | while read -r ref; do
    if [ -e "$SKILL_PATH/$ref" ]; then
      echo "REF_EXISTS: $ref"
    else
      echo "REF_MISSING: $ref"
    fi
  done
fi

EXECUTABLE_SCRIPTS=$(find "$SKILL_PATH/scripts" -type f 2>/dev/null)
echo ""
echo "=== SCRIPT EXECUTABILITY ==="
if [ -z "$EXECUTABLE_SCRIPTS" ]; then
  echo "NO_SCRIPTS: true"
else
  echo "$EXECUTABLE_SCRIPTS" | while read -r s; do
    REL=$(echo "$s" | sed "s|$SKILL_PATH/||")
    if [ -x "$s" ]; then
      echo "EXECUTABLE: $REL"
    else
      echo "NOT_EXECUTABLE: $REL"
    fi
  done
fi

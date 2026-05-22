#!/usr/bin/env bash
# Anti-pattern: unquoted variables -> word splitting and glob expansion
# Source: design-doc.md §2.4
# Why it breaks: a path with spaces becomes multiple arguments; a value
# containing '*' expands to every filename in the current directory.
# stop.sh fed an unquoted PID file path can delete the wrong things.

set -euo pipefail

echo "ANTI-PATTERN: unquoted variables"
echo "WHY: word splitting and globbing silently mutate command arguments."
echo
echo "DEMO 1: a path with spaces becomes multiple args."
path="dir with spaces/file.txt"
printf '  unquoted:   argv = ['
for a in $path; do printf '%s|' "$a"; done
echo ']'
printf '  quoted:     argv = ['
for a in "$path"; do printf '%s|' "$a"; done
echo ']'

echo
echo "DEMO 2: a variable containing '*' is glob-expanded in this dir."
pattern='*'
echo "  unquoted echo \$pattern -> $(echo $pattern | head -c 100)..."
echo "  quoted   echo \"\$pattern\" -> $pattern"

echo
echo "Imagine: rm -rf \$BUILD_DIR/target  -- if BUILD_DIR is empty, that becomes 'rm -rf /target'."

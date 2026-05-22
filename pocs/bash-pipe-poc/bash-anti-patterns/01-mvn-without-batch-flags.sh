#!/usr/bin/env bash
# Anti-pattern: invoking mvn without -B -ntp
# Source: design-doc.md §2.3 (skill anti-patterns), countered by G1
# Why it breaks: without -B (batch) mvn may prompt; without -ntp it emits
# per-byte transfer progress with carriage returns and ANSI colors.
# In a non-PTY Bash tool that captures (not streams) stdout the
# transcript becomes huge noise and the model sees nothing until EOF.

set -euo pipefail

echo "ANTI-PATTERN: mvn without -B -ntp"
echo "WHY: interactive prompts can hang; progress output floods the captured transcript."
echo
echo "DEMO: what 'mvn package' tends to emit when stdout is captured (not a TTY):"
echo "---"
printf 'Downloading from central: https://repo.maven.apache.org/.../foo-1.0.jar\r'
printf 'Progress (1): 12/256 kB\r'
printf 'Progress (1): 64/256 kB\r'
printf 'Progress (1): 128/256 kB\r'
printf 'Progress (1): 256/256 kB\n'
printf '\033[1;34m[INFO]\033[0m \033[1;32mBUILD SUCCESS\033[0m\n'
echo "---"
echo
echo "Every \\r line was a separate write that the model has to ingest."
echo "With -B -ntp the same step emits one line: '[INFO] BUILD SUCCESS'."

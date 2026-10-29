---
name: reproduce
description: Turn a bug report into a minimal failing reproduction before any fix. Use when the user reports a bug, asks to reproduce it or runs /debug:reproduce.
---

# Reproduce

Reproduce the bug described in the arguments: `$ARGUMENTS`.

1. Restate the bug as expected behavior versus actual behavior.
2. Find the smallest input that triggers it.
3. Write the smallest failing test or command that shows the actual behavior.
4. Run it and show that it fails for the reason described, not for an unrelated reason.

Do not fix the bug. Report the reproduction command, its output and the line where the behavior diverges.

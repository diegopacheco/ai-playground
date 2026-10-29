---
name: review
description: Review a file, a diff or a folder for correctness bugs, risky patterns and missing tests. Use when the user asks to review code or runs /code:review.
---

# Review

Review the target given in the arguments: `$ARGUMENTS`. When no target is given, review `git diff HEAD`.

1. Read the whole target before judging it.
2. Look for correctness bugs first: wrong logic, off by one, null handling, unhandled errors, resource leaks, races.
3. Then look for risky patterns: hardcoded secrets, unvalidated input, swallowed exceptions.
4. Then look for missing tests on the logic that matters.

Report a numbered list. Each item has the file and line, the problem in one sentence, and the fix in one sentence. Order by severity. When nothing is wrong, say so plainly.

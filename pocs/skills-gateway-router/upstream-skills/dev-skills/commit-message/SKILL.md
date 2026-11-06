---
name: commit-message
description: Writes a conventional commit message for the staged git changes.
---

Request: $ARGUMENTS

1. Run `git diff --staged`. If it is empty, tell the user nothing is staged and stop.
2. Write one conventional commit message: `<type>(<scope>): <summary>`, where type is feat, fix, docs, test, refactor or chore.
3. Keep the summary under 72 characters, in the imperative mood, and describe what the diff changes, not how.
4. Answer with the message only. Never run `git commit` yourself.

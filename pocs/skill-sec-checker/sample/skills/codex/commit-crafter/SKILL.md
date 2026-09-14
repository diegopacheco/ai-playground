---
name: commit-crafter
description: Writes a conventional commit message from the staged diff. Use when the user asks for a commit message.
allowed-tools: [Bash(git diff:*), Bash(git log:*)]
---

# Commit Crafter

1. Run `git diff --staged` and read every hunk.
2. Run `git log --oneline -10` to match the tone of recent messages.
3. Write one subject line under 72 characters in the form `type(scope): summary`.
4. Add a body only when the change needs a reason that the diff does not show.

Print the message. The user commits it.

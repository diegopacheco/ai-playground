---
name: fix-git-history
description: Rewrite poor git commit messages into a clear conventional-commit history. Reads the last N commits with their diffs, proposes a clean conventional message for each by reading what the code actually changed, then opens a light-theme review UI where the user approves all or each message individually before the history is rewritten. Use when the user runs /fix-git-history or asks to clean up, rewrite, normalize, or fix commit messages or git history into a conventional-commit format.
---

# fix-git-history

Turn a messy commit history into clean conventional-commit messages. The user reviews and approves every change in a light-theme web UI before anything is rewritten.

The script does the mechanical work (reading commits, serving the UI, rewriting history). You do the thinking: read each commit's diff and write a message that says, clearly and simply, what the change does.

## Steps

1. Confirm the current directory is a git repository and ask which one if it is ambiguous. Warn the user that this rewrites commit hashes for the chosen range. The working tree must be clean (no staged or unstaged changes to tracked files); if it is dirty, tell the user to commit or stash first and stop.

2. Collect the commits to review:

   ```
   python3 ~/.claude/skills/fix-git-history/fix_git_history.py collect <repo-dir> <work-dir>/commits.json [count]
   ```

   `count` defaults to 100. Use a `<work-dir>` of `.fix-git-history` inside the target repo. The script writes one JSON file with, per commit: full hash, short hash, author, date, the current (old) subject and body, the list of changed files with added/removed line counts, and a truncated unified diff.

3. Read `commits.json`. For every commit, look at the diff and the changed files and write a new message. Rules for the new message:
   - Conventional commit format: `type(scope): summary`. Scope is optional; use it only when one area clearly owns the change.
   - Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
   - Pick the type from what the diff does, not from the old message (the old message is usually wrong or empty).
   - The summary is lower-case, imperative, no trailing period, and short. Say what changed, derived from the code — not "update" or "wip".
   - Only add a body when the change genuinely needs one; keep it to one or two lines.

   Write your proposals to `<work-dir>/suggestions.json` as a JSON array of objects:

   ```json
   [{"hash": "<full hash>", "type": "feat", "message": "feat(task): add status filtering to task search", "reason": "adds a status query param and a findByStatus query"}]
   ```

   `message` is the full new subject line. `type` must match the prefix of `message`. `reason` is one short line shown in the UI so the user understands the suggestion. Include one object per commit you want to change; commits you leave out keep their original message.

4. Open the review UI:

   ```
   python3 ~/.claude/skills/fix-git-history/fix_git_history.py serve <repo-dir> <work-dir>/commits.json <work-dir>/suggestions.json [port]
   ```

   The script merges the two files, serves a light-theme page on `http://localhost:<port>` (it picks a free port if none is given) and opens it. Tell the user to review there: each card shows the old message, the suggested new one (editable), the changed files and the reason. They can toggle each commit on or off, edit any message, use **Approve all** / **Reject all**, then click **Apply approved**.

5. When the user clicks **Apply approved**, the page posts the approved messages back to the server. The server first creates a backup branch `backup/pre-fix-git-history-<timestamp>` pointing at the current HEAD, then rewrites the messages of the approved commits with `git filter-branch --msg-filter`, leaving the trees and parents untouched. The page shows the result and the backup branch name. Report the backup branch to the user and tell them how to undo: `git reset --hard <backup-branch>`.

## Notes

- The script uses only the Python standard library and the `git` already on the machine. Do not install anything.
- Only the approved commits change. Rewriting messages changes commit hashes for the approved commits and every commit after them, which is why the backup branch exists.
- Do not run this on a shared branch that others have already pulled unless the user understands the rewritten hashes will diverge from the remote.
- Merge commits are collected but their messages are usually left alone; only suggest a new message for a merge commit if it clearly needs one.
- If `collect` reports zero commits, confirm the path is a git repository with history before retrying.

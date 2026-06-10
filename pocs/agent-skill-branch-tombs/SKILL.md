---
name: branch-tombs
description: Builds a branch graveyard for a git repository. Reads every local and remote branch, records who last touched it and when, computes age, ahead/behind, and whether it is merged into the trunk, then renders a light-theme website that shows the stale branches as headstones and lists the ones that are safe to delete. Use when the user runs /branch-tombs or asks about stale, dead, abandoned, or old branches, branch cleanup, or which branches are safe to prune.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# Branch Graveyard

When invoked, you find the branches a repository has stopped touching and render a light-theme website that buries them. Each stale branch becomes a headstone showing its last-touch date and the author who last touched it. The report also lists the stale branches that are already merged into the trunk — the ones safe to delete — with the exact git command to remove each.

## Global Context
- User request / scope: $ARGUMENTS — empty, a number of days (the staleness threshold), or a GitHub repository to clone and analyze
- Engine: `scripts/branch_tombs.py` (zero third-party dependencies)
- Template: `assets/template.html`
- Output: `branch-graveyard-report/index.html` and `branch-graveyard-report/data.json` in the current directory

## Rules
- The numbers come from git refs only (`git for-each-ref`, `git rev-list`, `git merge-base`). Never invent or estimate ages, authors, or merge status. The engine produces every number deterministically.
- Read-only against the analyzed repository. The skill only writes inside `branch-graveyard-report/`, plus a temporary clone folder in GitHub mode that it deletes afterward.
- Do not add comments to any command you run.
- Never run the delete commands the report suggests. The report shows them for the user to copy; the skill never prunes anything.
- If the repository has no branches past the threshold, say the graveyard is empty instead of fabricating one.

## Step 1 — Choose the target

The scan runs against either the **current repository** or a **GitHub repository** the skill clones into a temporary folder. Decide from `$ARGUMENTS`:

- **`$ARGUMENTS` is a GitHub repository** — a URL like `https://github.com/owner/repo`, `git@github.com:owner/repo.git`, or the `owner/repo` shorthand → GitHub mode. Go to Step 2.
- **`$ARGUMENTS` is a number** (e.g. `60`) → current-repo mode with that staleness threshold in days. Skip to Step 3.
- **`$ARGUMENTS` is empty** → current-repo mode, default threshold of 30 days. Skip to Step 3.

If both a number and a GitHub repo are given, use the number as the threshold for that repo.

## Step 2 — Clone the GitHub repository (GitHub mode only)

A branch graveyard needs every remote branch, so clone with all branches but no working blobs to keep it fast:

```bash
tmp_root="$(mktemp -d)"
name="$(basename "<URL>" .git)"
git clone --no-checkout --filter=blob:none "<URL>" "$tmp_root/$name"
```

If the clone fails (bad URL, private repo with no access, no network), report the git error to the user and stop.

## Step 3 — Run the analysis engine

In current-repo mode, confirm a git repository first; GitHub mode skips this because the fresh clone is already one:

```bash
git rev-parse --show-toplevel
```

If this fails, tell the user the current directory is not a git repository and stop.

The engine detects the trunk (`origin/HEAD`, else `main`, else `master`, else the current branch), lists every branch under `refs/heads` and `refs/remotes`, collapses each local/remote pair into one branch, and for each computes the last-touch date, the author who made that commit, the age in days, ahead/behind versus the trunk, and whether it is merged. It then renders the website by injecting the data into the template. It always writes `branch-graveyard-report/` into the current working directory and finds its own template, so invoke it by its installed absolute path.

- **Current-repo mode** — pass the threshold from `$ARGUMENTS` if the user gave one, otherwise omit it:

  ```bash
  python3 "$HOME/.claude/skills/branch-tombs/scripts/branch_tombs.py" $ARGUMENTS
  ```

- **GitHub mode** — run the engine inside the clone with a subshell so the report still lands in the user's current directory, then copy it out and delete the clone:

  ```bash
  ( cd "$tmp_root/$name" && python3 "$HOME/.claude/skills/branch-tombs/scripts/branch_tombs.py" <days-if-any> )
  cp -R "$tmp_root/$name/branch-graveyard-report" .
  rm -rf "$tmp_root"
  ```

Notes:
- `python3` is required. If it is missing, tell the user to install Python 3.
- The threshold is a number of days; a branch older than it is counted as a grave. Default is 30.

## Step 4 — Open the report

```bash
open branch-graveyard-report/index.html
```

On Linux use `xdg-open branch-graveyard-report/index.html` instead.

## Step 5 — Summarize for the user

The engine prints a text summary to stdout. Relay the key signals in your reply:
- The repo (and, in GitHub mode, which repo was cloned) and the trunk branch.
- How many branches were scanned and how many are stale.
- How many are **safe to bury** (stale and already merged) versus **unmerged graves** (stale but holding work not in the trunk — these are the ones to review before deleting).
- The oldest branch (name, age, author) and the "gravekeeper" — the author with the most stale branches.
- The path to the rendered report.

Keep it short. The website carries the detail; your reply points the user at the oldest graves and the safe-to-delete count.

## How the website reads

- **The graveyard** — one headstone per stale branch, oldest first. Each stone shows the branch name, its last-touch date, the author who last touched it, how many days cold it is, and whether it is merged (green) or still holds unmerged commits (red). The headstone's top edge is colored by age tier: stale, abandoned, or ancient.
- **Safe to bury** — the stale branches already merged into the trunk, each with a copy-to-clipboard `git branch -d` / `git push --delete` command, plus a "copy all" button. Deleting these loses nothing unique.
- **All branches** — a sortable, filterable table of every branch with author, last touch, age, ahead/behind, location (local / remote / both), merged status, and tier.

## Staleness model (for reference)

Age is days since the branch tip's commit. Tiers: **active** (≤30 days), **stale** (≤90), **abandoned** (≤365), **ancient** (>365). A branch counts as a grave when its age exceeds the threshold (default 30 days) and it is not the trunk. A grave is **safe** when it is fully merged into the trunk, and an **unmerged grave** when it still has commits ahead of the trunk.

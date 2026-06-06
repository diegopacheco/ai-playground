---
name: bus-factor
description: Builds a knowledge-risk heatmap of a git repository. Runs git blame across tracked files, finds the ones owned by a single person, computes a bus factor per file, directory, and author, and renders a light-theme website that shows where knowledge is concentrated. Use when the user runs /bus-factor or asks where the project's knowledge risk, single points of failure, or single-owner code lives.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# Bus-Factor Map

When invoked, you measure how concentrated knowledge of a codebase is in single people, then render a light-theme website that shows the risk visually. The "bus factor" of a file is the smallest number of authors who together own more than half of its surviving lines. A bus factor of 1 means one person could leave and take the only knowledge of that file with them.

## Global Context
- User request / scope: $ARGUMENTS — empty, a local path to scope to, or a GitHub repository to clone and analyze
- Engine: `scripts/bus_factor.py` (zero third-party dependencies)
- Template: `assets/template.html`
- Output: `bus-factor-report/index.html` and `bus-factor-report/data.json` in the current directory

## Rules
- The numbers come from `git blame` only. Never invent or estimate ownership, line counts, or risk scores. The engine produces every number deterministically.
- Read-only against the analyzed repository. The skill only writes inside `bus-factor-report/`, plus a temporary clone folder in GitHub mode that it deletes afterward.
- Do not add comments to any command you run.
- If the engine prints nothing useful (no tracked source files), say so plainly instead of fabricating a report.

## Step 1 — Choose the target

The scan runs against either the **current repository** or a **GitHub repository** the skill clones into a temporary folder. Decide which from `$ARGUMENTS`:

- **`$ARGUMENTS` is a GitHub repository** — a URL like `https://github.com/owner/repo`, `git@github.com:owner/repo.git`, or the `owner/repo` shorthand → GitHub mode. Go to Step 2.
- **`$ARGUMENTS` is a local path** (e.g. `src/`, `services/api`) → current-repo mode scoped to that path. Skip to Step 3.
- **`$ARGUMENTS` is empty** — ask the user which they want with `AskUserQuestion`, two options:
  - **Current repository (Recommended)** — analyze the whole repo in the working directory. Skip to Step 3.
  - **A GitHub repo** — the user supplies the clone URL (they can paste it via "Other"). Then go to Step 2.

## Step 2 — Clone the GitHub repository (GitHub mode only)

Clone the **full history** into a temporary folder named after the repo. Full history is mandatory — a shallow clone collapses every line into one commit and destroys the blame attribution the whole analysis depends on.

```bash
tmp_root="$(mktemp -d)"
name="$(basename "<URL>" .git)"
git clone "<URL>" "$tmp_root/$name"
```

If the clone fails (bad URL, private repo with no access, no network), report the git error to the user and stop. Large repositories take longer to clone and to blame.

## Step 3 — Run the analysis engine

In current-repo mode, confirm a git repository first; GitHub mode skips this because the fresh clone is already one:

```bash
git rev-parse --show-toplevel
```

If this fails, tell the user the current directory is not a git repository and stop.

The engine enumerates tracked files with `git ls-files`, runs `git blame --line-porcelain -w -M -C` on each source file, counts surviving lines per author, and computes the per-file bus factor, per-author sole ownership, per-directory rollups, and a repo-level grade. It then renders the website by injecting the data into the template. It always writes `bus-factor-report/` into the current working directory and finds its own template, so invoke it by its installed absolute path.

- **Current-repo mode** — pass `$ARGUMENTS` as an optional subdirectory path, or omit it to scan the whole repository:

  ```bash
  python3 "$HOME/.claude/skills/bus-factor/scripts/bus_factor.py" $ARGUMENTS
  ```

- **GitHub mode** — run the engine inside the clone with a subshell so the report still lands in the user's current directory, then copy it out and delete the clone:

  ```bash
  ( cd "$tmp_root/$name" && python3 "$HOME/.claude/skills/bus-factor/scripts/bus_factor.py" )
  cp -R "$tmp_root/$name/bus-factor-report" .
  rm -rf "$tmp_root"
  ```

Notes:
- The engine resolves `assets/template.html` relative to its own file. The rendered `bus-factor-report/index.html` is self-contained, so removing the clone afterward does not affect it.
- `python3` is required. If it is missing, tell the user to install Python 3.
- On a very large repository the blame pass can take a while. If the user wants it faster, suggest scoping to a subdirectory, for example `/bus-factor src/`.

## Step 4 — Open the report

The report is now at `bus-factor-report/index.html` in the current working directory. Open it:

```bash
open bus-factor-report/index.html
```

On Linux use `xdg-open bus-factor-report/index.html` instead.

## Step 5 — Summarize for the user

The engine prints a text summary to stdout. Relay the key signals in your reply:
- The repo (and, in GitHub mode, which repo was cloned) and knowledge grade (A best, F worst).
- The repo bus factor.
- How many files are single-owned and what share of the code that is.
- The person with the biggest exposure (most lines solely owned) — the "if they leave" risk.
- The top knowledge-risk files.
- The path to the rendered report.

Keep it short. The website carries the detail; your reply points the user at the highest-risk findings.

## How the website reads

- **Heatmap** — a treemap where every box is a file. Box area is lines of code, box color is risk (green shared, red single-owner). This is the at-a-glance knowledge map.
- **If they leave** — authors ranked by how much code they solely own. The widest bars are the people whose departure would hurt most.
- **Risk by directory** — lines-of-code weighted risk per top-level directory, worst first.
- **All files** — a sortable, filterable table of every analyzed file with owner, share, authors, bus factor, lines, last touch, and risk.

## Risk model (for reference)

Per file, on a 0–100 scale:
- Owner concentration: the top author's share of surviving lines contributes up to 60.
- Spread: bus factor 1 adds 25, bus factor 2 adds 12, three or more adds 0.
- Staleness: time since the file's surviving lines last changed adds up to 15 (capped at two years).

A file is counted as single-owned when its top author holds at least 80% of the lines and its bus factor is 1. The repo grade is the lines-of-code weighted average risk mapped to A through F.

# codebase-roast-map

## Goal

Build a Claude Code and Codex skill that reads a repository and produces a funny visual pain map.

The skill finds complex files, risky ownership, stale areas, weak tests, churn hotspots, large files, bug-heavy paths, and folders that look expensive to maintain.

## Install Flow

`install.sh` asks where to install:

- Codex
- Claude Code
- Both

`uninstall.sh` asks the same question and removes installed files from the selected target.

Codex files are installed into:

- `~/.agents/skills/codebase-roast-map`

Claude Code files are installed into:

- `~/.claude/skills/codebase-roast-map`
- `~/.claude/commands/roast.md`
- `~/.claude/commands/roast-map.md`

## Commands

Claude Code installs these slash commands:

- `/roast`
- `/roast-map`

Codex CLI uses skills instead of custom slash commands. In Codex, use `/skills` and pick `codebase-roast-map`, or mention `$codebase-roast-map` in the prompt.

For Codex, ask:

- `$codebase-roast-map roast this repo`
- `$codebase-roast-map open the roast map`

The roast report prints a terminal report with ranked pain points.

The roast map generates `.roast-map/index.html`, `.roast-map/data.json`, `.roast-map/summary.md`, and opens the local UI when the operating system allows it.

## Signals

The scanner uses local data only:

- File tree
- Git history
- Recent commit subjects
- File size
- Line count
- Nesting
- Function-like blocks
- Test file proximity
- Contributor count
- Last touched date

## Score

Each file receives an explainable score from 0 to 100.

Signals increase the score:

- High line count
- High nesting
- Many function-like blocks
- High churn
- Many contributors
- Old last touched date
- Bug-related commit subjects
- Missing nearby tests
- Suspicious markers such as TODO, FIXME, and HACK

The score is not a replacement for engineering judgment. It is a fast triage lens.

## UI

The UI acts like a repo city map:

- Folders are districts
- Files are blocks
- Hotter colors mean higher risk
- Block size reflects file size and score
- Click opens a detail panel
- Search filters paths
- Layer buttons switch between overall risk, churn, complexity, stale code, ownership, tests, and bug risk

The UI is a static HTML page with inline CSS and JavaScript. It needs no server.

## Humor

The tone is funny but evidence-based.

The roast text must refer to concrete signals. It should never attack people.

## Constraints

- No network access
- No external libraries
- No source mutation during scan
- No aliases for misspelled commands
- Keep installation simple
- Keep generated output disposable

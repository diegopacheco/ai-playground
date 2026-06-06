# Design Doc — Bus-Factor Map Skill

## 1. Overview

`/bus-factor` is a Claude Code skill that turns a repository's git history into a **knowledge-risk heatmap**. It runs `git blame` across every tracked source file, works out who actually owns the surviving lines, computes a *bus factor* per file, per directory, and per author, and renders a self-contained light-theme website that shows — at a glance — where the project's knowledge is dangerously concentrated in a single person.

The name comes from the old engineering thought experiment: *how many people would have to be hit by a bus before the project is in serious trouble?* A bus factor of 1 for a file means one person could leave tomorrow and take the only working knowledge of that code with them.

## 2. Problem Statement

Every team has files that "only Sarah understands" and directories nobody dares touch since the author left. This risk is invisible until it bites — during an incident at 3am, during a reorg, or the week someone resigns. Teams have no cheap, repeatable way to *see* where that risk lives.

The raw data already exists in git. Nobody looks at it this way because turning thousands of `git blame` outputs into a single legible picture is tedious. This skill does exactly that and nothing else.

## 3. Goals

- Measure ownership from real git data — `git blame`, never estimates.
- Compute a per-file bus factor: the smallest set of authors owning more than half the surviving lines.
- Identify single-owner files (one person holds ≥80% of the lines and bus factor is 1).
- Roll the risk up to directories, authors, and the whole repo.
- Quantify the "if they leave" exposure per person: how much code walks out the door with them.
- Render it all as a light-theme, dependency-free website whose centerpiece is a treemap heatmap.
- Run in seconds-to-low-minutes on a normal repo with zero third-party dependencies.

## 4. Non-Goals

- Not a productivity or performance metric. Owning code is not a virtue or a vice; concentration is just risk.
- Not a blame tool in the social sense — it measures knowledge distribution, not fault.
- No code-quality, complexity, or churn analysis (other skills cover that).
- No multi-identity reconciliation (one human with several git emails counts as several authors). Out of scope for v1.
- No server, no database, no build step. The output is one static HTML file plus its JSON.

## 5. Architecture

### 5.1 Pipeline

```
git ls-files            git blame (per file)        aggregate                 render
+----------------+     +---------------------+     +-----------------+     +------------------+
| enumerate      |---->| --line-porcelain    |---->| per-file risk   |---->| inject JSON into |
| tracked files, |     | -w -M -C            |     | per-dir rollup  |     | template.html    |
| filter source  |     | count lines/author  |     | per-author sole |     | -> index.html    |
+----------------+     +---------------------+     | repo grade      |     +------------------+
                                                    +-----------------+
```

Two components:

| Component | File | Responsibility |
|---|---|---|
| Engine | `scripts/bus_factor.py` | Pure Python stdlib. Walks git, computes every number, writes `data.json`, injects it into the template to produce `index.html`. Deterministic. |
| View | `assets/template.html` | Self-contained light-theme page. Inline CSS + vanilla JS, no libraries, no build. Reads the injected data and renders the visuals. |

The skill (`SKILL.md`) only orchestrates: pick the target (current repo or a GitHub repo to clone), confirm/clone the repository, run the engine by its installed absolute path, open the report, and relay the headline numbers. **No number in the report originates from the model** — the engine produces them all, so two runs on the same commit give identical results.

### 5.5 Target selection: current repo or GitHub clone

The skill works on the current repository or on a GitHub repository it clones on demand. The choice is driven by the argument: a URL (`https://github.com/owner/repo`, `git@github.com:owner/repo.git`) or `owner/repo` shorthand triggers GitHub mode; a local path scopes the current repo; an empty argument makes the skill ask the user which they want.

In GitHub mode the repo is cloned into a temporary folder with **full history** — a shallow clone would attribute every surviving line to a single squashed commit and make the bus factor meaningless. The engine runs inside the clone (so the git root resolves cleanly and avoids `/tmp` symlink edge cases), but it always writes `bus-factor-report/` to the user's current working directory; the rendered HTML is self-contained, so the clone is deleted as soon as the report exists. No change to the engine was needed — it already discovers the git root from where it runs and writes its output to the current directory.

### 5.2 Why a deterministic engine, not the model

Bus factor is arithmetic over `git blame`. Letting the model "read git log and estimate" would make the output non-reproducible and prone to drift on large repos. The engine guarantees the numbers are real and the report is auditable via `data.json`. The model's value is orchestration and pointing the user at the worst findings.

### 5.3 The blame pass

For each candidate file:

```
git blame --line-porcelain -w -M -C HEAD -- <file>
```

- `--line-porcelain` emits a full header (including `author` and `committer-time`) for every line, so line-by-line ownership is a simple parse.
- `-w` ignores whitespace-only changes, so reformatting does not transfer ownership.
- `-M` detects moves within a file; `-C` detects copies from other files — both keep ownership with the real author instead of whoever moved the code.

Lines are tallied per author. The newest `committer-time` across surviving lines gives the file's recency (staleness input).

### 5.4 File filtering

To keep the blame pass fast and meaningful, the engine skips noise:

- **Directories**: `.git`, `node_modules`, `vendor`, `dist`, `build`, `target`, `out`, `bin`, `obj`, `__pycache__`, `.next`, `.venv`, `venv`, `coverage`, `.gradle`, `.idea`, `.terraform`, and similar.
- **Lock/generated files**: `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `bun.lock`, `Cargo.lock`, `go.sum`, `poetry.lock`, `composer.lock`, `Gemfile.lock`, etc.
- **Binary/asset extensions**: images, fonts, archives, compiled artifacts, media, `*.min.js`, `*.min.css`, `*.map`.

Everything else tracked by git is treated as source.

## 6. The Risk Model

Each file gets a risk score on a 0–100 scale, built from three interpretable parts:

| Component | Range | Meaning |
|---|---|---|
| Owner concentration | 0–60 | `top_author_share × 60`. One person owning all lines is the dominant risk. |
| Spread (bus factor) | 0 / 12 / 25 | Bus factor 1 → +25, bus factor 2 → +12, three or more → +0. |
| Staleness | 0–15 | Time since the surviving lines last changed, ramping linearly and capping at two years. Stale single-owner code is worse — the owner may be gone or have forgotten it. |

Total is capped at 100. Tiers for the heatmap colors:

| Risk | Tier | Color |
|---|---|---|
| 0–24 | low | green |
| 25–49 | medium | yellow-green |
| 50–74 | high | orange |
| 75–100 | critical | red |

**Bus factor of a file** = walk authors from largest line-share down, counting how many it takes to cross 50% of the file's lines. **Repo bus factor** = the same computation over every author's total surviving lines across all analyzed files.

**Single-owned file** = top author share ≥ 80% *and* bus factor = 1. These feed the per-author "if they leave" exposure (count of files and total lines that person solely owns).

**Repo grade** = lines-of-code weighted average risk, mapped: `<15 → A`, `<30 → B`, `<45 → C`, `<60 → D`, else `F`. Weighting by LOC means a 2000-line single-owner module hurts the grade far more than a 5-line config file.

## 7. The Website

A single self-contained `index.html`, light theme, no external requests, no build. Five regions:

1. **Header + grade badge** — repo name, scope, generation time, and the A–F knowledge grade.
2. **Summary cards** — repo bus factor, single-owner file count and percentage, contributor count, files analyzed, and the biggest single-person exposure.
3. **Knowledge-risk heatmap (centerpiece)** — a squarified treemap where every box is a file: **area = lines of code, color = risk**. The largest files are labeled; hovering any box shows owner, share, authors, bus factor, lines, last touch, and risk tier. This is the "see it in one glance" view.
4. **If they leave** — authors ranked by lines of code they solely own, as horizontal bars. The widest bar is the person whose departure would hurt most.
5. **Risk by directory** — LOC-weighted risk per top-level directory, worst first.
6. **All files** — a sortable, filterable table of every analyzed file, color-coded by tier, defaulting to worst-risk-first.

### 7.1 Why a treemap

A treemap is the canonical way to show a file tree where two things matter at once: *how big* (area) and *how risky* (color). A flat list loses the size signal; a bar chart loses the structure. The squarified variant keeps boxes close to square so small high-risk files stay clickable and the whole picture reads as a heat map. It is implemented in ~40 lines of vanilla JS (Bruls et al. squarify), so there is no charting dependency.

## 8. Diagrams

Pipeline (engine):

```
[ repo @ HEAD ] --ls-files--> [ candidate files ] --blame--> [ line counts / author / file ]
                                                                         |
                                                                         v
                                            [ per-file risk + bus factor ]
                                              |                |              |
                                              v                v              v
                                     [ directory rollup ] [ author sole ] [ repo grade ]
                                              \________________|______________/
                                                               v
                                                     data.json + index.html
```

Heatmap encoding:

```
   +-----------------+-------+----+      area  = lines of code
   |   auth.ts       | ui.ts | db |      color = knowledge risk
   |   RED  bf=1     | YEL   |ORG |
   +--------+--------+-------+----+      green  = shared, many owners
   | api.ts | cfg    |  utils.ts  |      red    = single owner, bus factor 1
   | ORG    | GRN    |    GREEN   |
   +--------+--------+------------+
```

## 9. Invocation

```
/bus-factor                                scan the whole current repository
/bus-factor src/                           scan only a subdirectory (faster on large repos)
/bus-factor services/api                   scan a specific module
/bus-factor https://github.com/owner/repo  clone a GitHub repo (full history) and scan it
/bus-factor owner/repo                     same, using the owner/repo shorthand
```

With no argument the skill asks whether to scan the current repository or a GitHub repository. Output is written to `bus-factor-report/index.html` and `bus-factor-report/data.json` in the user's current directory, and opened in the browser. In GitHub mode the temporary clone is removed once the report is rendered.

## 10. File Structure

```
~/.claude/skills/bus-factor/          (installed)
  SKILL.md                            orchestration
  scripts/bus_factor.py               the analysis engine
  assets/template.html                the website template

agent-skill-bus-factor/               (this POC)
  design-doc.md                       this document
  README.md                           overview + screenshots
  install.sh                          copies the skill into ~/.claude/skills/bus-factor
  uninstall.sh                        removes it
  SKILL.md  scripts/  assets/         the skill payload
  printscreens/                       rendered report screenshots
```

## 11. Suggestions and Improvements

### What makes this skill valuable
- **Zero config, zero dependencies** — Python stdlib and git, nothing to install.
- **Reproducible** — same commit, same numbers, every time; `data.json` is auditable.
- **Reads in one glance** — the treemap turns thousands of blame lines into a single picture a manager or a new hire understands immediately.
- **Actionable** — the "if they leave" panel names the exact people and the exact files to pair-program, document, or review first.

### Potential enhancements (future)
- **Identity mapping** via a `.mailmap`-style file so one human's multiple emails collapse to one author.
- **Diff mode** — compare two commits and show whether knowledge risk improved or got worse over a release.
- **CI gate** — fail a build (or post a comment) when the repo grade drops below a threshold or a critical file gains a second risk factor.
- **Recency-only mode** — weight by how recently each author touched a file to surface "owner has gone quiet" cases.
- **Drill-down** — click a treemap box to open the file's per-author line breakdown.

## 12. Critiques and Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| `git blame` measures surviving lines, not understanding | Someone who wrote code then left still shows as owner | Staleness term down-weights nothing here but flags old single-owned code; pair with the `last touch` column |
| One human, many emails counts as many authors | Can understate concentration | Document it; add `.mailmap` support in v2 |
| Blame is O(files); large monorepos are slow | Long runs on huge repos | Scope to a subdirectory argument; noise directories are skipped by default |
| Generated code that is checked in inflates a single author | False high-risk hotspots | Extend the deny lists; treat as a known caveat |
| Line count is a rough proxy for knowledge | A 10-line crypto core may matter more than a 1000-line fixture | Risk also reflects concentration and staleness, not size alone; size only drives treemap area |
| Vendored code without history | Skipped or attributed to the importer | Deny lists exclude common vendor paths |
| GitHub mode clones full history over the network | Slow / impossible for huge or private repos without access | Full clone is required for correct blame; fall back to running the skill inside an already-checked-out copy |

## 13. Success Criteria

- Runs on any git repo with `python3` and produces `index.html` + `data.json` with no third-party packages.
- Every number in the report is derivable from `git blame` on the same commit.
- The treemap renders with correct area (LOC) and color (risk), and tooltips show real per-file data.
- The "if they leave" ranking matches the per-author sole-ownership totals in `data.json`.
- `install.sh` places the skill under `~/.claude/skills/bus-factor`; `uninstall.sh` removes it cleanly.

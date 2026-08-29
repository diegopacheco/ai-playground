# Design Doc

Codebase Inventory Skill — `/inventory`.

Scans a codebase three times, cross-checks every claim, and produces one
self-contained light-themed searchable HTML report with seven tabs.

## Architecture Principles

* Facts are collected by code, judgement is written by the agent.
  `scan.py` never guesses; the agent never counts.
* Three passes, each with a different job: collect, verify, attack.
  A claim that survives all three ships; anything else is dropped.
* The renderer is a gate, not a printer. `render.py` refuses to emit a report
  when a file path does not exist or a required section is incomplete.
* One output file. `index.html` embeds its own CSS, JS, data and diagram.
* Prompts live in `prompts/`, never inside the code.

## Features

* Triple scan with an explicit verification and an adversarial pass.
* Seven tabs: Architecture, Modules, Schema, Committers, Tech Debt,
  Observability, Tests.
* Global search across modules, tech debt, tables, committers and tests.
* Hand-drawn architecture diagram generated from a node/edge graph with a
  layered layout that guarantees no overlap.
* Report written to a temp directory; the temp directory path is printed at
  the top of the report itself.
* Installs into Claude Code, Codex, or both; the installer asks.

## Overall Diagram

```
/inventory
    |
    v
+-------------------+     facts.json      +--------------------+
|    scan.py        | ------------------> |   agent 3 passes   |
|  git, files, tests|                     | pass1 collect      |
|  logs, schema, td |                     | pass2 verify       |
+-------------------+                     | pass3 adversarial  |
                                          +--------------------+
                                                   |
                                            analysis.json
                                                   v
                                          +--------------------+
                                          |     render.py      |
                                          |  validate + merge  |
                                          +--------------------+
                                                   |
                                                   v
                                    $TMPDIR/inventory-<repo>-<ts>/index.html
```

## TradeOffs

| Decision | Gain | Cost |
| --- | --- | --- |
| Python stdlib only | Runs anywhere python3 runs, no install step | Hand-written parsers for SQL/JPA/test detection |
| Agent writes the prose | Explanations a human can act on | Non-deterministic wording between runs |
| Renderer validates and fails | No invented file paths reach the report | A weak analysis pass fails the run instead of degrading |
| Single self-contained HTML | Mail it, open it offline, no server | Report file is large (~200KB) |
| Avatars from github.com/<user>.png | Real faces, zero API key | Needs network; falls back to inline initials SVG |
| Temp dir output | Never dirties the scanned repo | Report is lost on reboot unless copied |

## Decisions

1. **Three passes are separate prompt files, not one big prompt.** Each pass
   sees the previous output and has a single job. A merged prompt drifts into
   summarizing instead of checking.
2. **`scan.py` emits raw evidence, not conclusions.** Line numbers, counts and
   paths. This is what pass 2 verifies against.
3. **Modules come from build files first, directories second.** A `pom.xml`,
   `package.json`, `go.mod` or `Cargo.toml` is a stronger module signal than a
   folder name.
4. **Schema tab is conditional.** When no schema is discovered the tab is not
   rendered rather than rendered empty.
5. **Table size is reported as row-count evidence or `n/a`.** Live database
   size is not reachable from a static scan and is never invented.
6. **Tech debt is capped at 10 per module.** Ranked by evidence count so the
   list stays actionable.
7. **No metaphors in generated prose.** Enforced in the prompts and spot
   checked by pass 3; con explanations are capped at 2 lines, fixes at 3.
8. **The installer asks Claude / Codex / both.** Copying to a tool the user
   does not use is noise.

## How to Run

```bash
./install.sh            # asks: Claude Code, Codex, or both
/inventory              # in any repository
/inventory path/to/sub  # scan a subdirectory
./uninstall.sh          # asks which target to remove
```

## How to Test

```bash
./test.sh
```

Runs the scanner against the bundled fixture, renders a report from a fixture
analysis, and asserts: valid JSON facts, all seven tab sections present when
data exists, schema tab absent when no schema exists, every referenced path
resolves, and the validator rejects a malformed analysis.

## All REST Endpoints

None. This is a command line skill, there is no service.

---
name: pass-2-verify
params: $facts_json, $repo_root, $report_dir
---

Pass 2 of 3 — VERIFY.

Re-scan the codebase at `$repo_root` against the draft you wrote at
`$report_dir/analysis.json`. Treat the draft as a stranger's claims.

## What to do

For every claim in the draft, open the file again and check it.

1. **Paths.** Every `path` in `modules[].files`, `techDebt[].examples`,
   `schema.tables[].source` and `tests.issues[].path` must exist. Run a real
   check. Delete or correct every path that does not resolve.
2. **Line numbers.** Open each tech-debt example at its line and confirm the
   line still shows the problem described. Correct the number or drop it.
3. **Counts.** Compare every number you wrote against `$facts_json`. Test
   counts, table counts, commit counts and log counts come from the scan, not
   from memory. Replace mismatches with the scan value.
4. **Coverage.** Any module in the facts with source files but no entry in
   `modules` must be added now, or explicitly folded into another module by
   listing its directory under that module's `paths`.
5. **Edges.** For each architecture edge, name the import, client or config
   entry that proves it. Drop edges you cannot prove.
6. **Committers.** Confirm each committer's "works on" claim against the
   directories the scan recorded for that author. Do not describe work you
   cannot see in the data.

Rewrite `$report_dir/analysis.json` in place with the corrections applied and
append to `verification` a one-line note per correction you made, in the form
`fixed: <what was wrong> -> <what it is>`.

A pass that reports zero corrections on a codebase of any size is a pass that
did not run. Show your checks.

---
name: pass-1-collect
params: $facts_json, $repo_root, $report_dir
---

Pass 1 of 3 — COLLECT.

You are inventorying the codebase at `$repo_root`.
The mechanical scan already ran and its raw evidence is at `$facts_json`.
Read it in full before you read any source file.

Your job in this pass is to turn evidence into a first draft. Do not verify
yet, do not polish. Write down what you believe is true and where you saw it.

## What to do

1. Read `$facts_json`.
2. Read the real files it points at. Read the top 3 files of every module,
   every schema file, every alert/dashboard file, and a sample of test files
   for each detected test type. Reading the scan output is not enough.
3. Produce `$report_dir/analysis.json` following `prompts/schema.md` exactly.

## Rules for this pass

* Every module in `modules` must map to a module id present in the facts.
* Every file path you write must be a path that exists, relative to the repo
  root, copied from the facts or from a file you actually opened.
* Never invent a table, a committer, a test count or a metric. If the facts do
  not contain it and you cannot read it, leave the field out.
* Architecture nodes must be things that exist in the repo (a module, a data
  store, an external service found in config). Edges must be justified by an
  import, a client class, a config entry or a build dependency.
* Tech debt items must each carry at least one real example with a file path
  and a line number.
* Write the prose now, at draft quality. Pass 3 will judge the writing.

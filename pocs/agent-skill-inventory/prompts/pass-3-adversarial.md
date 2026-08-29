---
name: pass-3-adversarial
params: $facts_json, $repo_root, $report_dir
---

Pass 3 of 3 — ADVERSARIAL.

Your job is to break `$report_dir/analysis.json`. Assume it is wrong and find
where. Only what survives this pass reaches the report.

## Attack it on four fronts

**1. Invention.** For each statement, ask: which file proves this? If nothing
in the repo proves it, delete the statement. Suspect anything that sounds like
general advice about software rather than a fact about this repository.

**2. Contradiction.** Cross-read the sections. A module called well tested in
tab 2 while tab 7 shows zero tests for it is a contradiction. A tech-debt item
that a module lists as a pro is a contradiction. Resolve every one, keeping
the side the evidence supports.

**3. Writing quality.** Enforce these limits and rewrite anything that misses:
* Module `description`: 3 to 5 lines, plain language, no metaphors.
* Every con `why`: at most 2 lines, says what goes wrong for a real person.
* Every con `fix`: at most 3 lines, a concrete action on named code.
* Exactly 5 pros and 5 cons per module, and per schema section when present.
* No metaphors anywhere. No "spaghetti", "glue", "swiss cheese", "band-aid",
  "smell", "rot", "battle-tested", "under the hood". Say the mechanism.
* No marketing words. No hedging that carries no information.

**4. Usefulness.** A con that no one can act on is not a con. A tech-debt item
without a fix is a complaint. Cut anything a maintainer could not start on
this week.

## Finish

* Rank `techDebt` per module by evidence count, keep the top 10 per module.
* Order `committers` by commit count, descending.
* Append to `verification` one line per change, prefixed `cut:` or `rewrote:`.
* Set `verification.passes` to 3.

Write the final `$report_dir/analysis.json`. Then run the renderer. If the
renderer rejects the file, fix the reported problem and run it again. Do not
edit the renderer to make it pass.

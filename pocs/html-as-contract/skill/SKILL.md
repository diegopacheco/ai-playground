---
name: html-as-contract
description: Author and verify engineering rules as a single HTML contract. Use when the user runs /html-as-contract to review, accept, or suggest rules, or /html-as-contract check to scan code and regenerate the compliance report tab. Callable by a human or by another agent.
---

# html-as-contract

A single `contract.html` holds engineering rules (tab 1) and a live compliance
report (tab 2). It is dual-reader: a human reads the rendered light-theme page,
an agent reads the same markup and writes the report back into it.

Dispatch on the argument after the command.

## Locate the contract

- The contract is `contract.html` in the current working directory, or the path
  the user names.
- The HTML scaffold lives next to this skill at `template.html`.

## Region ownership

- `#rules` (tab 1) is human-authored. In every mode except review/suggest it is
  read-only. Never rewrite it during a check.
- `#compliance-report` (tab 2) is agent-owned and replaced wholesale on check.
- The chrome and `#agent-protocol` are fixed. Do not touch them.

## Mode: review (no argument) — `/html-as-contract`

1. If `contract.html` is missing, copy `template.html` to `contract.html`.
2. Read the rules in `#rules` and present them.
3. Let the user accept, edit, or remove rules. Apply accepted changes inside
   `#rules` only.
4. Do not touch `#compliance-report`.

## Mode: suggest — `/html-as-contract suggest`

1. Ensure `contract.html` exists; scaffold from `template.html` if missing.
2. Scan the codebase and propose candidate rules, each with category, severity,
   check type, rationale, and a good/bad code pair.
3. Present each proposal. Add only accepted ones into `#rules`.

## Mode: check — `/html-as-contract check`

1. Read `contract.html` and follow the embedded `#agent-protocol`.
2. For each `.rule` in `#rules`, scan the scope (default project root; the sample
   scans `sample/src`).
3. Decide `pass` / `fail` / `unknown`, honoring `data-check`: `static` is a
   deterministic pattern match, `semantic` is judgment and must state confidence.
4. Replace only the contents of `#compliance-report`: the summary stats, a
   `.meta` line with generated-at and scope, and one `.report-item` per rule with
   `data-rule-id` and `data-status`, citing each violation as `file:line`.
5. Do not modify `#rules`. This mode is non-interactive and safe for a human or
   an agent to call.

## Rules region format

Each rule is:

```
<article class="rule" data-rule-id="..." data-category="testing|design|code"
         data-severity="error|warning|info" data-check="static|semantic">
  <h3>title <span class="badge error">error</span> <span class="badge check">static</span></h3>
  <p class="statement">the rule</p>
  <p class="rationale">why it matters</p>
  <div class="pair">
    <div class="col"><p class="label good-label">good</p><pre class="good">...</pre></div>
    <div class="col"><p class="label bad-label">bad</p><pre class="bad">...</pre></div>
  </div>
</article>
```

Rules are grouped under `<h2 class="category">` headings per category.

## Report region format

```
<div class="summary">
  <div class="stat pass"><div class="num">N</div><div class="lbl">passed</div></div>
  <div class="stat fail"><div class="num">N</div><div class="lbl">failed</div></div>
  <div class="stat unknown"><div class="num">N</div><div class="lbl">unknown</div></div>
</div>
<p class="meta">Generated &lt;iso-timestamp&gt; · scope &lt;path&gt;</p>
<div class="report-item" data-rule-id="..." data-status="pass|fail|unknown">
  <h4>title <span class="status-tag">fail</span></h4>
  <div class="violation"><code>file:line</code> — short reason</div>
</div>
```

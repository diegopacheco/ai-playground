---
name: bug-cataloger
description: Find bugs that were encountered and solved in prior Claude Code or Codex sessions for the current project, then generate a self-contained searchable light-theme HTML bug report. Use when the user asks for a bug catalog, solved-problem history, prior-session bug report, recurring failure inventory, or a BUG to Solution report for the current repository.
---

# Bug Cataloger

Build a report from recorded evidence in prior sessions for the current project.

## Workflow

1. Detect the active agent before reading sessions:
   - Use `claude` when running in Claude Code.
   - Use `codex` when running in Codex.
   - Run `python3 scripts/collect_sessions.py --project "$PWD" --output /tmp/bug-cataloger-sessions.jsonl` from this skill directory. Add `--agent claude` or `--agent codex` only if automatic detection fails.
2. Treat the current folder as the project boundary. Record its containing Git repository for context without including sessions from sibling projects.
3. Review the collection summary. Stop with a clear message when no matching prior sessions exist.
4. Read the entire normalized transcript in chunks. Search for failure terms, tool failures, corrections, changed files, successful tests, and user confirmations. Inspect surrounding entries for every candidate.
5. Include a bug only when the sessions contain both the failure and credible evidence that the user resolved it, including changes applied through the agent. Do not infer a solution from an unverified suggestion.
6. Merge repeated occurrences of the same root cause. Exclude unfinished work, feature requests, environment setup without a defect, and failures unrelated to the current project.
7. Write `/tmp/bug-cataloger-findings.json` with this shape:

```json
{
  "agent": "Codex",
  "project": "project-name",
  "bugs": [
    {
      "bug": ["Failure symptom", "Affected behavior", "Root cause"],
      "solution": ["Change applied", "Why it fixed the issue", "Verification observed"]
    }
  ]
}
```

8. Keep every `bug` and `solution` array between 3 and 7 short lines. Remove secrets, credentials, personal data, raw prompts, hidden reasoning, and unrelated file paths.
9. Run `python3 scripts/render_report.py --input /tmp/bug-cataloger-findings.json --output-dir "$PWD"` from this skill directory.
10. Return the absolute report path and the number of cataloged bugs.

## Evidence Rules

- Prefer explicit error output, a concrete code or configuration change, and passing verification.
- Accept user confirmation as verification when no automated check exists.
- Preserve technical specificity without copying long transcript passages.
- Produce an empty report when no solved bugs meet the evidence threshold.
- Never modify the project while cataloging its history.

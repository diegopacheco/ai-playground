---
name: skill-evaluator
description: This skill should be used when the user asks to "evaluate a skill", "review skill quality", "score my skill", "check skill best practices", "rate my skills", "evaluate all skills", "compare skills", or wants to assess skill quality across criteria like clarity, token efficiency, anti-cheating, quality gates, determinism, scope discipline, error recovery, observability, and idempotency.
---

# Skill Evaluator

Evaluate skills for Claude Code and Fox Codex across 9 quality criteria, producing a scored pass/fail report with actionable recommendations.

## Trigger

This skill is triggered by the `/skill-evaluator` command.

## Phase 1: Skill Discovery

Scan the following locations for installed skills:

| Location | Platform |
|---|---|
| `~/.claude/skills/` | Claude Code |
| `~/.claude/plugins/*/skills/` | Claude Code plugins |
| `~/.codex/skills/` | Fox Codex |

For each location, find directories containing a `SKILL.md` file.

Present the discovered skills in a table:

```
| # | Skill | Platform | Path |
|---|---|---|---|
```

Ask the user using AskUserQuestion: "Evaluate ALL skills or select specific ones?"

If user selects specific ones, present a multi-select list using AskUserQuestion with `multiSelect: true`.

If a path argument is provided, skip discovery and evaluate that single skill directly.

## Phase 2: Evaluation

For each selected skill, read its `SKILL.md` and all files in its directory tree.

Evaluate the skill against all 9 criteria. For each criteria, analyze the skill content and assign a score from 0 to 10.

### Criteria 1: Direct and Clear (0-10)

Evaluate how direct, precise, and clear the skill is.

Look for:
- Filler words: "please", "kindly", "basically", "essentially", hedging
- Vague verbs: "handle", "manage", "process" instead of "parse", "validate", "write"
- Redundant repetition of the same idea
- Long compound sentences that should be short
- Preamble and throat-clearing introductions
- Missing success criteria on steps

Score >= 6 to PASS.

### Criteria 2: Token Efficiency (0-10)

Evaluate how efficiently the skill uses context window tokens.

Look for:
- SKILL.md word count (ideal 500-2000, penalty above 3000)
- Duplicate information across SKILL.md and references/
- Over-explanation of things the LLM already knows
- Verbose paragraphs that should be tables or lists
- Whether heavy content is in references/ (progressive disclosure)

Score <= 5 to PASS (lower is better, fewer tokens wasted).

### Criteria 3: Anti-Cheating (0-10)

Evaluate trustworthiness guardrails.

Look for:
- Verification steps that check outputs are real
- Rules forbidding hardcoded or fabricated results
- Correctness validation before accepting results
- Rollback mechanisms for failed validations
- Explicit anti-cheat rules section
- Requirements for real execution (not just "looks right")
- Audit trail (logs, outputs, diffs as evidence)

Score >= 6 to PASS.

### Criteria 4: Reinforcing Quality Gates (0-10)

Evaluate whether the skill blocks progress until quality bars are met.

Look for:
- "Do not continue until X passes" rules
- Tests must pass before next phase
- Build must succeed before proceeding
- Checkpoints between phases
- Rules saying gates cannot be skipped
- Defined failure handling (fix, retry, stop)
- Progressive strictness across phases

Score >= 6 to PASS.

### Criteria 5: Determinism (0-10)

Evaluate reproducibility across runs.

Look for:
- Pinned versions for dependencies and tools
- Fixed seeds for random operations
- Avoidance of non-deterministic choices
- Same input producing same output guarantee
- Independence from transient system state

Score >= 6 to PASS.

### Criteria 6: Scope Discipline (0-10)

Evaluate whether the skill stays in its lane.

Look for:
- Rules against modifying code outside scope
- Only creating strictly necessary files
- Not adding comments or docstrings unless asked
- Not refactoring surrounding code
- Minimal footprint policy
- No feature creep beyond what was requested

Score >= 6 to PASS.

### Criteria 7: Error Recovery (0-10)

Evaluate failure handling.

Look for:
- Step failure detection
- Retry logic for transient failures
- Rollback mechanism for partial work
- User notification on errors
- No silent failures policy
- Graceful degradation to safe state

Score >= 6 to PASS.

### Criteria 8: Observability (0-10)

Evaluate evidence production.

Look for:
- Output artifacts (files, logs, reports)
- Before/after diffs
- Command output capture
- Progress indicators during multi-step work
- Audit trail for post-execution review

Score >= 6 to PASS.

### Criteria 9: Idempotency (0-10)

Evaluate re-run safety.

Look for:
- Safe to run twice without duplicating work
- No conflicting state on second run
- Detection of prior runs
- Clean re-entry without corruption
- No side-effect accumulation

Score >= 6 to PASS.

## Phase 3: Report Generation

For each evaluated skill, produce a results table:

```
# Skill Evaluation: [skill-name]
Date: YYYY-MM-DD
Path: /path/to/skill/

| Criteria             | Score | Result | Summary                                       |
|----------------------|-------|--------|-----------------------------------------------|
| Direct and Clear     | X/10  | PASS   | [short summary of what is good or to improve] |
| Token Efficiency     | X/10  | PASS   | [short summary of what is good or to improve] |
| Anti-Cheating        | X/10  | PASS   | [short summary of what is good or to improve] |
| Quality Gates        | X/10  | PASS   | [short summary of what is good or to improve] |
| Determinism          | X/10  | PASS   | [short summary of what is good or to improve] |
| Scope Discipline     | X/10  | PASS   | [short summary of what is good or to improve] |
| Error Recovery       | X/10  | PASS   | [short summary of what is good or to improve] |
| Observability        | X/10  | PASS   | [short summary of what is good or to improve] |
| Idempotency          | X/10  | PASS   | [short summary of what is good or to improve] |

Overall: X.X/10 | PASS/FAIL (list failing criteria if FAIL)
```

Pass/fail rules:
- All criteria PASS at score >= 6, EXCEPT Token Efficiency which PASSES at score <= 5
- A skill PASSES overall only if ALL 9 criteria pass individually

If multiple skills were evaluated, append a ranking table:

```
| Rank | Skill | Overall | Result | Weakest Criteria |
|------|-------|---------|--------|------------------|
```

Write the full report to `eval-report.md` in the current working directory.

## Rules

- Read every file in the skill directory before scoring
- Run `scripts/evaluate.sh` on each skill to collect static metrics before LLM analysis
- Never invent or assume content that is not in the skill files
- Every score must have a concrete justification in the Summary column
- If a skill has zero anti-cheat rules, Anti-Cheating score is 0
- If a skill has zero quality gates, Quality Gates score is 0
- Token Efficiency score is inverted: lower score = better efficiency = PASS

## Additional Resources

### Scripts
- **`scripts/evaluate.sh`** - Static analysis script that collects word count, file structure, frontmatter validation, and checks for common patterns

### Reference Files
- **`references/scoring-rubric.md`** - Detailed scoring rubric with score guides for each criteria
- **`references/best-practices-checklist.md`** - Checklist of best practices to evaluate against

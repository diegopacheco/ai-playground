# Skill Evaluator - Design Document

## Problem Statement

Skills for Claude Code and Fox Codex are written manually with no automated quality assessment.
Skill authors have no way to know if their SKILL.md follows best practices, if trigger descriptions
are effective, if progressive disclosure is properly implemented, or if the skill will actually
activate when users need it. This leads to skills that are bloated, poorly triggered, or ineffective.

## Goal

Build a skill that evaluates other skills across multiple quality dimensions, produces a scored
report, and provides actionable recommendations for improvement.

## Target Platforms

- Claude Code (skills in `~/.claude/skills/` and plugin skills)
- Fox Codex (compatible skill format)

## How It Works

The skill evaluator receives a path to a skill directory (or scans all installed skills) and
performs a multi-dimensional evaluation, producing a scored report.

## Evaluation Criteria

### 1. Direct and Clear (0-10)

How direct, precise, and clear is the skill? Does it go straight to the point or does it
ramble with filler, vague language, and unnecessary padding?

| Check | What It Evaluates |
|---|---|
| No filler words | Avoids "please", "kindly", "basically", "essentially", hedging language |
| Precise instructions | Each instruction tells exactly what to do, no ambiguity |
| No redundancy | Same idea is not repeated in different words across the skill |
| Short sentences | Prefers short, direct sentences over long compound ones |
| Concrete verbs | Uses "parse", "validate", "write" not "handle", "manage", "process" |
| No preamble | Jumps straight into what matters, no throat-clearing introductions |
| Clear success criteria | Each step has a clear definition of done |

**Score guide:**
- **10**: Every sentence carries information. Zero filler. Reads like a spec.
- **7-9**: Mostly direct, minor verbosity in places.
- **4-6**: Mix of clear instructions and vague hand-waving.
- **1-3**: Walls of text, unclear what to actually do.
- **0**: Completely vague, no actionable content.

### 2. Token Efficiency (0-10)

How efficiently does the skill use context window tokens? Does it pack maximum value per
token or waste tokens on fluff, duplication, and over-explanation?

| Check | What It Evaluates |
|---|---|
| Word count vs information density | Ratio of actionable content to total words |
| No duplicate information | Same fact is not stated in SKILL.md and also in references/ |
| Progressive disclosure | Heavy content lives in references/, SKILL.md stays lean |
| No over-explanation | Does not explain obvious things the LLM already knows |
| Compact formatting | Uses tables and lists instead of verbose paragraphs |
| SKILL.md body size | Ideal: 500-2000 words. Penalty above 3000 words |
| Reference file usage | Large context moved to references/ loaded on demand |

**Score guide:**
- **10**: Under 1500 words, every word earns its place. Perfect progressive disclosure.
- **7-9**: Lean skill, minor bloat. Good use of references/.
- **4-6**: Noticeable waste. Repeated content. Could cut 30-50% without losing meaning.
- **1-3**: Bloated. Paragraphs of obvious information. Everything in one file.
- **0**: Massive wall of text, most of it noise.

### 3. Anti-Cheating (0-10)

How trustworthy is the skill? Does it have guardrails against the LLM producing BS,
hallucinations, fake results, or skipping real work? Does it enforce honesty?

| Check | What It Evaluates |
|---|---|
| Verification steps | Requires checking outputs are real (not hardcoded, not fabricated) |
| No fake results | Explicitly forbids hardcoding outputs or returning static data |
| Correctness validation | Mandates validation before accepting results |
| Rollback on failure | Has a mechanism to undo work that fails validation |
| Explicit anti-cheat rules | States what counts as cheating and forbids it |
| Real execution required | Forces actual execution (run tests, run benchmarks) not just "looks right" |
| Audit trail | Produces evidence that work was actually done (logs, outputs, diffs) |

**Score guide:**
- **10**: Comprehensive anti-cheat rules. Every output verified. Cannot fake results.
- **7-9**: Good guardrails, most outputs verified, minor gaps.
- **4-6**: Some verification but easy to skip. LLM could cut corners undetected.
- **1-3**: Minimal verification. Trusts LLM output blindly.
- **0**: No anti-cheat at all. Accepts whatever the LLM produces.

### 4. Reinforcing Quality Gates (0-10)

Does the skill enforce gates that block progress until a quality bar is met? Does it ensure
tests pass, code compiles, benchmarks run, or validation succeeds before moving forward?

| Check | What It Evaluates |
|---|---|
| Gate-before-proceed | Explicit "do not continue until X passes" rules |
| Tests must pass | Requires test execution and pass before moving to next phase |
| Build must succeed | Code must compile/build before proceeding |
| Validation checkpoints | Has checkpoints between phases where quality is verified |
| No skipping gates | Explicitly states gates cannot be skipped or bypassed |
| Failure handling | Defines what happens when a gate fails (fix, retry, stop) |
| Progressive gates | Gates get stricter as the skill progresses through phases |

**Score guide:**
- **10**: Cannot proceed without passing every gate. Clear failure handling at each gate.
- **7-9**: Strong gates on critical paths. Minor steps might lack enforcement.
- **4-6**: Has some checkpoints but they are suggestions, not hard blocks.
- **1-3**: Mentions quality but does not enforce it. Easy to skip ahead.
- **0**: No gates. Linear flow with no quality checks.

### 5. Determinism (0-10)

Does the skill produce consistent results across runs? Does it pin versions, use fixed seeds,
avoid non-deterministic choices? Or does every run give a different outcome?

| Check | What It Evaluates |
|---|---|
| Pinned versions | Dependencies, tools, and runtimes have explicit versions |
| Fixed seeds | Random operations use fixed seeds or deterministic alternatives |
| No non-deterministic choices | Avoids arbitrary picks that change between runs |
| Reproducible outputs | Same input produces same output every time |
| Environment independence | Does not depend on transient system state |

**Score guide:**
- **10**: Fully reproducible. Pinned versions, fixed seeds, identical output every run.
- **7-9**: Mostly deterministic, minor variance in non-critical outputs.
- **4-6**: Some runs differ. Unpinned versions or random choices present.
- **1-3**: Results vary significantly between runs.
- **0**: Every run produces different results.

### 6. Scope Discipline (0-10)

Does the skill stay in its lane? Does it avoid doing things the user did not ask for
(adding comments, refactoring nearby code, creating extra files)? Or does it over-engineer
and gold-plate?

| Check | What It Evaluates |
|---|---|
| No unsolicited changes | Does not modify code outside its scope |
| No extra files created | Only creates files that are strictly necessary |
| No comments added | Does not add comments or docstrings unless asked |
| No refactoring | Does not refactor surrounding code |
| Minimal footprint | Changes only what is needed to complete the task |
| No feature creep | Does not add features beyond what was requested |

**Score guide:**
- **10**: Surgical precision. Only touches what was asked. Zero side effects.
- **7-9**: Stays focused, minor unnecessary additions.
- **4-6**: Some scope creep. Adds things not requested.
- **1-3**: Regularly does unsolicited work. Over-engineers.
- **0**: Rewrites everything it touches.

### 7. Error Recovery (0-10)

When something fails mid-skill, does it handle it gracefully? Does it retry, rollback, or
ask the user? Or does it silently continue with broken state?

| Check | What It Evaluates |
|---|---|
| Failure detection | Detects when a step fails |
| Retry logic | Retries transient failures before giving up |
| Rollback mechanism | Can undo partial work when a step fails |
| User notification | Informs the user when something goes wrong |
| No silent failures | Never continues silently after an error |
| Graceful degradation | Falls back to a safe state on failure |

**Score guide:**
- **10**: Every failure detected, handled, and communicated. Rollback available.
- **7-9**: Good error handling on critical paths. Minor gaps.
- **4-6**: Some errors caught, others ignored. Partial rollback.
- **1-3**: Failures mostly ignored. Continues with broken state.
- **0**: No error handling. Silent failures everywhere.

### 8. Observability (0-10)

Does the skill produce evidence of what it did? Logs, diffs, outputs, before/after
comparisons? Or is it a black box where you just hope it worked?

| Check | What It Evaluates |
|---|---|
| Output artifacts | Produces files, logs, or reports showing what was done |
| Before/after diffs | Shows what changed |
| Execution logs | Captures command outputs and results |
| Progress indicators | Reports progress during multi-step operations |
| Audit trail | Leaves a trail that can be reviewed after the fact |

**Score guide:**
- **10**: Full audit trail. Every action logged. Before/after diffs. Nothing hidden.
- **7-9**: Good visibility. Most actions produce evidence.
- **4-6**: Some outputs visible, others are fire-and-forget.
- **1-3**: Minimal evidence. Hard to tell what happened.
- **0**: Black box. No logs, no diffs, no outputs.

### 9. Idempotency (0-10)

Can you run the skill twice and get the same result without breaking anything? Or does
running it again duplicate work, create conflicts, or corrupt state?

| Check | What It Evaluates |
|---|---|
| Safe to re-run | Running twice does not duplicate files, entries, or changes |
| No conflicting state | Second run does not conflict with first run artifacts |
| Detects prior runs | Checks if work was already done before doing it again |
| Clean re-entry | Can resume or restart without corrupting previous results |
| No side-effect accumulation | Repeated runs do not pile up side effects |

**Score guide:**
- **10**: Perfectly idempotent. Run 10 times, same result as running once.
- **7-9**: Mostly idempotent, minor artifacts on re-run.
- **4-6**: Re-running causes some duplicates or requires manual cleanup.
- **1-3**: Re-running breaks things or creates conflicts.
- **0**: Cannot be run twice without manual intervention.

## Scoring

### Per-Criteria Score
Each criteria produces a score from 0 to 10.

### Overall Score
Average of all 9 criteria (0 to 10):

| Criteria | Weight |
|---|---|
| Direct and Clear | ~11% |
| Token Efficiency | ~11% |
| Anti-Cheating | ~11% |
| Reinforcing Quality Gates | ~11% |
| Determinism | ~11% |
| Scope Discipline | ~11% |
| Error Recovery | ~11% |
| Observability | ~11% |
| Idempotency | ~11% |

### Grade Mapping

| Score | Grade | Meaning |
|---|---|---|
| 9-10 | A | Production-ready, follows all best practices |
| 7-8 | B | Good quality, minor improvements possible |
| 5-6 | C | Acceptable, several areas need attention |
| 3-4 | D | Below average, significant improvements needed |
| 0-2 | F | Fails basic quality checks, needs rewrite |

## Pass/Fail Rules

Each criteria has a pass/fail threshold:

| Criteria | Pass Condition | Rationale |
|---|---|---|
| Direct and Clear | Score >= 6 | Skill must be clear enough to be useful |
| Token Efficiency | Score <= 5 | Lower is better — fewer tokens wasted. Above 5 means bloated |
| Anti-Cheating | Score >= 6 | Must have minimum guardrails against BS |
| Reinforcing Quality Gates | Score >= 6 | Must enforce at least basic quality gates |
| Determinism | Score >= 6 | Results must be reproducible |
| Scope Discipline | Score >= 6 | Must not do unsolicited work |
| Error Recovery | Score >= 6 | Must handle failures gracefully |
| Observability | Score >= 6 | Must produce evidence of what it did |
| Idempotency | Score >= 6 | Must be safe to re-run |

A skill PASSES overall only if ALL 9 criteria pass individually.

## Output Table Format

The evaluator produces a single table per skill with all criteria, score, pass/fail, and a
short summary explaining what is good or what needs improvement.

```
# Skill Evaluation: [skill-name]
Date: YYYY-MM-DD
Path: /path/to/skill/

| Criteria             | Score | Result | Summary                                                 |
|----------------------|-------|--------|---------------------------------------------------------|
| Direct and Clear     | 8/10  | PASS   | Clear instructions, minor verbosity in phase 3          |
| Token Efficiency     | 3/10  | PASS   | Lean SKILL.md, good use of references/                  |
| Anti-Cheating        | 4/10  | FAIL   | No verification steps, outputs not validated             |
| Quality Gates        | 7/10  | PASS   | Tests must pass before deploy, no gate on build step     |
| Determinism          | 9/10  | PASS   | Pinned versions, fixed seeds, reproducible outputs       |
| Scope Discipline     | 6/10  | PASS   | Stays focused but adds one extra config file             |
| Error Recovery       | 5/10  | FAIL   | Catches build failures but ignores test timeouts         |
| Observability        | 7/10  | PASS   | Produces findings.md and report.md with diffs            |
| Idempotency          | 8/10  | PASS   | Safe to re-run, detects prior wave results               |

Overall: 6.3/10 | FAIL (Anti-Cheating and Error Recovery below threshold)
```

When evaluating multiple skills, produce one table per skill followed by a ranking table:

```
# Ranking

| Rank | Skill          | Overall | Result | Weakest Criteria    |
|------|----------------|---------|--------|---------------------|
| 1    | autobench      | 8.2/10  | PASS   | Token Efficiency    |
| 2    | json-formatter | 6.0/10  | FAIL   | Anti-Cheating       |
| 3    | workflow-skill | 4.5/10  | FAIL   | Quality Gates       |
```

## Modes of Operation

### Default Mode (Interactive)
When invoked with no arguments, the skill:

1. Scans `~/.claude/skills/` and `~/.codex/skills/` for all installed skills
2. Lists all found skills in a numbered table
3. Asks the user: "Evaluate ALL skills or select specific ones?"
4. If user picks "all" — evaluates every skill and produces a ranking table
5. If user picks "select" — presents a multi-select list for the user to choose which skills to evaluate

```
/skill-evaluator
```

### Single Skill Evaluation
Evaluate one skill by providing a path:
```
/skill-evaluator path/to/skill-name/
```

### Batch Evaluation
Evaluate all installed skills without prompting:
```
/skill-evaluator --all
```

### Compare Mode
Evaluate and rank multiple skills side by side:
```
/skill-evaluator --compare path/to/skill1 path/to/skill2
```

### Skill Discovery

The evaluator scans the following locations for installed skills:

| Location | Platform |
|---|---|
| `~/.claude/skills/` | Claude Code user skills |
| `~/.claude/plugins/*/skills/` | Claude Code plugin skills |
| `~/.codex/skills/` | Fox Codex user skills |

For each location, it finds directories containing a `SKILL.md` file and treats them as skills.

The discovery output looks like:

```
Found 12 installed skills:

| # | Skill              | Platform    | Path                                    |
|---|--------------------|-------------|-----------------------------------------|
| 1 | autobench          | Claude Code | ~/.claude/skills/autobench/             |
| 2 | json-formatter     | Claude Code | ~/.claude/skills/json-formatter/        |
| 3 | frontend-design    | Claude Code | ~/.claude/skills/frontend-design/       |
| 4 | workflow-skill     | Claude Code | ~/.claude/skills/workflow-skill/        |
| 5 | hook-development   | Plugin      | ~/.claude/plugins/.../hook-development/ |
| ...                                                                             |

Evaluate ALL or select specific skills? [all/select]
```

## Implementation Plan

### Phase 1: Core Evaluator
- SKILL.md for the skill-evaluator skill
- Evaluation logic as a structured workflow in SKILL.md
- scripts/evaluate.sh - bash script that performs static checks (file existence, word count, frontmatter validation)

### Phase 2: Content Analysis
- LLM-powered analysis for writing style, trigger quality, and actionable instructions
- Scoring engine that aggregates check results into dimension scores

### Phase 3: Report Generation
- Generate eval-report.md with full findings
- Support batch and compare modes

## Installation

### install.sh

Installs the skill-evaluator skill to both Claude Code and Fox Codex.

- Copies the `skill-evaluator/` directory into `~/.claude/skills/skill-evaluator/`
- Copies the `skill-evaluator/` directory into `~/.codex/skills/skill-evaluator/`
- Creates target directories if they do not exist
- Prints success/failure for each platform

### uninstall.sh

Removes the skill-evaluator skill from both Claude Code and Fox Codex.

- Removes `~/.claude/skills/skill-evaluator/`
- Removes `~/.codex/skills/skill-evaluator/`
- Prints success/failure for each platform

## Project Directory Structure

```
skill-evaluator/
├── design-doc.md
├── install.sh
├── uninstall.sh
├── README.md
└── skill-evaluator/
    ├── SKILL.md
    ├── scripts/
    │   └── evaluate.sh
    └── references/
        ├── scoring-rubric.md
        └── best-practices-checklist.md
```

## Key Design Decisions

1. **Bash + LLM hybrid** - Static checks (file exists, word count, frontmatter parsing) run via bash script for speed and determinism. Qualitative checks (writing style, trigger quality) use LLM analysis.

2. **Weighted scoring** - Trigger quality and content quality have the highest weight (25% each) because they most directly impact whether a skill actually works when needed.

3. **Actionable output** - Every finding includes a priority level (HIGH/MED/LOW) and a concrete fix suggestion, not just a pass/fail.

4. **No external dependencies** - The evaluator uses only bash, sed, grep, and wc. No Python, no npm packages.

5. **Platform agnostic** - Works for both Claude Code and Fox Codex skills since both follow the same SKILL.md format.

## Open Questions

1. Should the evaluator auto-fix simple issues (missing frontmatter fields, wrong person in description)?
2. Should there be a "strict mode" that fails on any non-A grade?
3. Should evaluation results be cached to track improvement over time?

# Scoring Rubric

## Pass/Fail Thresholds

| Criteria | Pass Condition |
|---|---|
| Direct and Clear | >= 6 |
| Token Efficiency | <= 5 (lower is better) |
| Anti-Cheating | >= 6 |
| Quality Gates | >= 6 |
| Determinism | >= 6 |
| Scope Discipline | >= 6 |
| Error Recovery | >= 6 |
| Observability | >= 6 |
| Idempotency | >= 6 |

## Direct and Clear

| Score | Description |
|---|---|
| 10 | Every sentence carries information. Zero filler. Reads like a spec. |
| 7-9 | Mostly direct, minor verbosity in places. |
| 4-6 | Mix of clear instructions and vague hand-waving. |
| 1-3 | Walls of text, unclear what to actually do. |
| 0 | Completely vague, no actionable content. |

Signals: filler words count, vague verb count, sentence length, redundancy.

## Token Efficiency

| Score | Description |
|---|---|
| 10 | Under 1500 words, every word earns its place. Perfect progressive disclosure. |
| 7-9 | Lean skill, minor bloat. Good use of references/. |
| 4-6 | Noticeable waste. Repeated content. Could cut 30-50% without losing meaning. |
| 1-3 | Bloated. Paragraphs of obvious information. Everything in one file. |
| 0 | Massive wall of text, most of it noise. |

Signals: word count, references/ usage, duplicate content, over-explanation.

## Anti-Cheating

| Score | Description |
|---|---|
| 10 | Comprehensive anti-cheat rules. Every output verified. Cannot fake results. |
| 7-9 | Good guardrails, most outputs verified, minor gaps. |
| 4-6 | Some verification but easy to skip. LLM could cut corners undetected. |
| 1-3 | Minimal verification. Trusts LLM output blindly. |
| 0 | No anti-cheat at all. Accepts whatever the LLM produces. |

Signals: verification steps, anti-cheat rules section, correctness validation, audit trail.

## Quality Gates

| Score | Description |
|---|---|
| 10 | Cannot proceed without passing every gate. Clear failure handling at each gate. |
| 7-9 | Strong gates on critical paths. Minor steps might lack enforcement. |
| 4-6 | Has some checkpoints but they are suggestions, not hard blocks. |
| 1-3 | Mentions quality but does not enforce it. Easy to skip ahead. |
| 0 | No gates. Linear flow with no quality checks. |

Signals: "must pass", "do not continue", "before proceeding", checkpoint mentions.

## Determinism

| Score | Description |
|---|---|
| 10 | Fully reproducible. Pinned versions, fixed seeds, identical output every run. |
| 7-9 | Mostly deterministic, minor variance in non-critical outputs. |
| 4-6 | Some runs differ. Unpinned versions or random choices present. |
| 1-3 | Results vary significantly between runs. |
| 0 | Every run produces different results. |

Signals: version pins, fixed seeds, reproducibility guarantees.

## Scope Discipline

| Score | Description |
|---|---|
| 10 | Surgical precision. Only touches what was asked. Zero side effects. |
| 7-9 | Stays focused, minor unnecessary additions. |
| 4-6 | Some scope creep. Adds things not requested. |
| 1-3 | Regularly does unsolicited work. Over-engineers. |
| 0 | Rewrites everything it touches. |

Signals: "only create", "do not modify", "minimal", "no extra" rules.

## Error Recovery

| Score | Description |
|---|---|
| 10 | Every failure detected, handled, and communicated. Rollback available. |
| 7-9 | Good error handling on critical paths. Minor gaps. |
| 4-6 | Some errors caught, others ignored. Partial rollback. |
| 1-3 | Failures mostly ignored. Continues with broken state. |
| 0 | No error handling. Silent failures everywhere. |

Signals: rollback, retry, revert, "on failure", "if fails", error handling mentions.

## Observability

| Score | Description |
|---|---|
| 10 | Full audit trail. Every action logged. Before/after diffs. Nothing hidden. |
| 7-9 | Good visibility. Most actions produce evidence. |
| 4-6 | Some outputs visible, others are fire-and-forget. |
| 1-3 | Minimal evidence. Hard to tell what happened. |
| 0 | Black box. No logs, no diffs, no outputs. |

Signals: report generation, findings logs, diff output, progress indicators.

## Idempotency

| Score | Description |
|---|---|
| 10 | Perfectly idempotent. Run 10 times, same result as running once. |
| 7-9 | Mostly idempotent, minor artifacts on re-run. |
| 4-6 | Re-running causes some duplicates or requires manual cleanup. |
| 1-3 | Re-running breaks things or creates conflicts. |
| 0 | Cannot be run twice without manual intervention. |

Signals: "already exists" checks, "skip if", prior run detection, re-run safety.

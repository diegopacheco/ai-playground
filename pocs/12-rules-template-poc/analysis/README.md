# Analysis — `CLAUDE.md` (12-rule template)

Opus 4.7 Analyis.

Target file: [`../CLAUDE.md`](../CLAUDE.md)

## Overall score

**67 / 100** — solid principles, weak enforcement, missing project context.

| Dimension | Score | Notes |
|---|---|---|
| Clarity | 8 / 10 | Rules are short, well-named, no jargon |
| Specificity | 6 / 10 | Some rules give concrete tests, others stay abstract |
| Observability (can the model self-check?) | 5 / 10 | Several rules can't be enforced without external state |
| Coverage | 6 / 10 | No project-specific context: stack, build, test, conventions |
| Conciseness | 9 / 10 | Tight; nothing redundant |
| Actionability | 7 / 10 | Most rules translate directly to behavior change |
| **Weighted total** | **67** | |

## Per-rule effectiveness

Scale: ★☆☆ low · ★★☆ medium · ★★★ high effectiveness for steering model behavior.

| # | Rule | Score | Why |
|---|---|---|---|
| 1 | Think Before Coding | ★★★ | Pushes assumption-naming and pushback; immediately actionable |
| 2 | Simplicity First | ★★★ | The "senior engineer" sniff test is a concrete check |
| 3 | Surgical Changes | ★★★ | Strongest scope-creep guard; observable in diffs |
| 4 | Goal-Driven Execution | ★★☆ | "Define success criteria" is good, but doesn't say *how* to verify |
| 5 | Use the model only for judgment calls | ★★★ | Sharp, rare-quality rule; prevents anti-patterns like LLM-as-router |
| 6 | Token budgets are not advisory | ★☆☆ | Numbers (4k/30k) are unobservable to the model in-session; aspirational |
| 7 | Surface conflicts, don't average them | ★★★ | Prevents the worst failure mode (silent blending) |
| 8 | Read before you write | ★★★ | Easy to follow, high payoff |
| 9 | Tests verify intent, not just behavior | ★★☆ | Right principle, but abstract — no example of intent-encoded test |
| 10 | Checkpoint after every significant step | ★★☆ | "Significant" undefined; risks over- or under-checkpointing |
| 11 | Match the codebase's conventions | ★★★ | Clear precedence (conformance > taste); escape hatch included |
| 12 | Fail loud | ★★★ | Excellent counter to "task complete" inflation |

Average per-rule effectiveness: **2.5 / 3** (≈ 83%).
The headline 67/100 drops below that because the file is also judged on what it *doesn't* cover.

## What works

- **Stated bias up front** ("caution over speed on non-trivial work") frames every rule.
- **Each rule has a why**, not just a what — the model can apply judgment to edge cases.
- **No redundancy** across rules; each carves a distinct failure mode.
- **Rules 3, 5, 7, 12 are unusually sharp** — they target failure modes most prompts miss.
- **Override clause** ("unless explicitly overridden") leaves room without forking the file.

## Gaps

1. **No project context.** Nothing about the stack, build command, test command, lint, or how to run the app. Compare to the global `~/.claude/CLAUDE.md`, which is concrete (podman, podman-compose, no comments, no sleeps > 1). The project file is pure principle.
2. **Rule 6 is unenforceable.** The model can't measure 4,000 tokens reliably. This rule will be either ignored or invoked theatrically. Either replace with an observable proxy (e.g. "if a response exceeds N lines of diff, stop and checkpoint") or remove.
3. **Rules 4, 9, 10 lack worked examples.** "Define success criteria", "tests encode why", "checkpoint after a significant step" — each would gain ★ with one concrete example.
4. **No examples of bad vs good output.** Even one before/after pair per rule would double its retention.
5. **No commit / PR conventions in the project file.** These exist globally but a project file is the right place to restate the load-bearing ones.
6. **No tie-break against the global `CLAUDE.md`.** When the user's global file and project file conflict, which wins? Rule 7 (surface conflicts) implicitly covers this but it's worth stating.
7. **"Significant step" is undefined** (Rule 10) — a model with a long task list may either checkpoint on every tool call or never.

## Highest-leverage edits

In order of expected impact:

1. Add a **Project context** section: stack, run command, test command, lint command, where state lives. One screen, no prose.
2. Replace Rule 6's token numbers with an observable rule, or move the budgets to a runtime-enforced hook.
3. Add one concrete example to Rules 4, 9, 10.
4. Add a one-line conflict-precedence rule: *project CLAUDE.md > user CLAUDE.md > model defaults*, with Rule 7 procedure if they truly clash.
5. Define "significant step" — e.g. "after any file write, test run, or branch operation".

## Verdict

A strong **principles** file — well above average for what most repos ship. As a **working** file for steering day-to-day tasks in this specific project, it's incomplete: it tells the model how to think but not what it's looking at. Pair it with a short project-context block and Rule 6 either gets teeth or gets cut, and the score jumps into the 80s.

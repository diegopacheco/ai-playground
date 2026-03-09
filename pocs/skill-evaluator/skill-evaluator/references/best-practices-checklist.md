# Best Practices Checklist

## Structure
- [ ] SKILL.md exists
- [ ] Valid YAML frontmatter with name and description
- [ ] Markdown body present and substantial
- [ ] All referenced files exist on disk
- [ ] Directories (references/, scripts/, assets/) used appropriately

## Trigger Description
- [ ] Uses third person ("This skill should be used when...")
- [ ] Contains at least 3 specific quoted trigger phrases
- [ ] No vague language without specifics

## Writing Style
- [ ] Imperative/infinitive form (verb-first instructions)
- [ ] No second person ("you should", "you need to")
- [ ] Short, direct sentences
- [ ] No filler words (please, kindly, basically, essentially)
- [ ] Concrete verbs (parse, validate, write) not vague (handle, manage, process)

## Token Efficiency
- [ ] SKILL.md body under 2000 words (ideal), under 3000 (acceptable)
- [ ] Heavy content in references/, not SKILL.md
- [ ] No duplicate information across files
- [ ] No over-explanation of LLM-known concepts
- [ ] Tables and lists preferred over paragraphs

## Anti-Cheating
- [ ] Has verification steps for outputs
- [ ] Forbids hardcoded or fabricated results
- [ ] Requires correctness validation
- [ ] Has rollback on failed validation
- [ ] Explicit anti-cheat rules section
- [ ] Requires real execution, not guessing
- [ ] Produces audit trail (logs, outputs, diffs)

## Quality Gates
- [ ] "Do not continue until X passes" rules
- [ ] Tests must pass before next phase
- [ ] Build must succeed before proceeding
- [ ] Checkpoints between phases
- [ ] Gates cannot be skipped or bypassed
- [ ] Failure handling defined (fix, retry, stop)

## Determinism
- [ ] Pinned versions for dependencies
- [ ] Fixed seeds for random operations
- [ ] No non-deterministic choices
- [ ] Same input produces same output
- [ ] No dependency on transient system state

## Scope Discipline
- [ ] Rules against modifying code outside scope
- [ ] Only creates strictly necessary files
- [ ] No unsolicited comments or docstrings
- [ ] No refactoring of surrounding code
- [ ] No feature creep

## Error Recovery
- [ ] Detects step failures
- [ ] Retry logic for transient failures
- [ ] Rollback for partial work
- [ ] User notification on errors
- [ ] No silent failures
- [ ] Graceful degradation

## Observability
- [ ] Produces output artifacts (files, logs, reports)
- [ ] Shows before/after diffs
- [ ] Captures command outputs
- [ ] Reports progress during multi-step work
- [ ] Leaves audit trail

## Idempotency
- [ ] Safe to run twice
- [ ] No conflicting state on re-run
- [ ] Detects prior runs
- [ ] Clean re-entry without corruption
- [ ] No side-effect accumulation

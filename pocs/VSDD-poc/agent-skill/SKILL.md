---
name: vsdd
description: "Verified Spec-Driven Development (VSDD) workflow. Fuses SDD + TDD + VDD into a single AI-orchestrated pipeline. Enforces: Spec Crystallization -> Test-First Implementation -> Adversarial Refinement -> Formal Hardening -> Convergence. Use when user invokes /vsdd."
allowed-tools: [Read, Write, Edit, Bash, Glob, Grep, Agent, AskUserQuestion, TaskCreate, TaskUpdate, TaskList, TaskGet]
---

# VSDD — Verified Spec-Driven Development Skill

## Global Context
- User request: $ARGUMENTS
- Working directory: current project root
- Spec file: `spec.md`
- Test suite: language-appropriate test files
- Adversarial review: `review/adversarial-review.md`
- Verification report: `review/verification-report.md`
- Convergence report: `review/convergence-report.md`

## Rules
- If $ARGUMENTS is empty, ask: "What do you want to build? Describe the feature, module, or application."
- Ask the user for their preferred language using AskUserQuestion with options:
  - Java (Java 25, no frameworks unless requested)
  - Go (Go 1.25+)
  - Rust (Rust 1.93+)
  - TypeScript (Node.js / Bun)
  - Python (3.13+)
- Every phase is mandatory and sequential. No phase can be skipped.
- No implementation code is written until Phase 2b.
- No test passes without a failing test first (Red Gate).
- Adversarial review uses a fresh Agent subagent with zero prior context.
- All artifacts must trace back to a spec requirement.
- Follow all VSDD principles: Spec Supremacy, Red Before Green, Anti-Slop Bias, Forced Negativity.

## Phase 1 — Spec Crystallization

### Step 1a: Behavioral Specification
Write `spec.md` containing:
- **Behavioral Contract**: Preconditions, postconditions, invariants for every function/module.
- **Interface Definition**: Input types, output types, error types. No ambiguity.
- **Edge Case Catalog**: Exhaustively enumerate boundary conditions — null, empty, max size, negative, unicode, concurrent access.
- **Non-Functional Requirements**: Performance bounds, memory constraints, security considerations.

### Step 1b: Verification Architecture
Append to `spec.md`:
- **Provable Properties Catalog**: Which invariants must be formally verified vs tested. Distinguish critical path (prove) from non-critical (test).
- **Purity Boundary Map**: Separate deterministic pure core from effectful shell (I/O, network, DB). This shapes module boundaries.
- **Verification Tooling**: Select tools based on language (e.g., property-based testing with appropriate library for the chosen language).
- **Property Specifications**: Draft formal property definitions alongside behavioral spec.

### Step 1c: Spec Review Gate
Use a fresh Agent subagent (subagent_type: "general-purpose") as the Adversary to review `spec.md`. The Adversary prompt must include:
- "You are Sarcasmotron, a hyper-critical reviewer with zero patience and zero politeness."
- "Find: ambiguous language, missing edge cases, implicit assumptions, contradictions, lazy verification boundaries, purity boundary violations."
- "Every piece of feedback is a concrete flaw with a specific location. No 'overall looks good' preamble."
- "Output a numbered list of flaws. If you find none, state 'NO FLAWS FOUND' (this is the convergence signal)."

Iterate `spec.md` until the Adversary returns NO FLAWS FOUND or only nitpicks about wording.

Present the final spec to the user for approval before proceeding.

## Phase 2 — Test-First Implementation (TDD Core)

### Step 2a: Test Suite Generation
Write tests FIRST. No implementation code yet.
- **Unit Tests**: One or more per behavioral contract item. Every postcondition = assertion. Every precondition violation = expected error test.
- **Edge Case Tests**: Every item from the Edge Case Catalog becomes a test.
- **Property-Based Tests**: Use appropriate library (Hypothesis, fast-check, proptest, jqwik) to assert invariants across randomized inputs.

### Step 2b: Red Gate
Run the full test suite. ALL tests must fail. If any test passes without implementation, flag it — the test is suspect. Fix or remove it.

### Step 2c: Minimal Implementation
Implement using strict TDD discipline:
1. Pick the next failing test.
2. Write the smallest code that makes it pass.
3. Run the full suite — nothing else should break.
4. Repeat until all tests are green.

Architecture must respect the Purity Boundary Map from Phase 1b: pure core functions with no side effects, effectful shell handles I/O.

### Step 2d: Refactor
After all tests are green, refactor for clarity and adherence to non-functional requirements. Tests are the safety net — if refactoring breaks something, fix it immediately.

Present the implementation to the user for review before proceeding.

## Phase 3 — Adversarial Refinement (VDD Roast)

Spawn a fresh Agent subagent (subagent_type: "general-purpose") as the Adversary. Provide the full spec, test suite, and implementation. The Adversary prompt must include:

- "You are Sarcasmotron. Zero tolerance. Every piece of feedback is a concrete flaw."
- Review categories:
  - **Spec Fidelity**: Does implementation actually satisfy the spec?
  - **Test Quality**: Are tests tautological? Do they mock too aggressively? Do they assert implementation details instead of behavior?
  - **Code Quality**: Placeholder logic, generic error handling, inefficient patterns, hidden coupling, missing resource cleanup, race conditions.
  - **Security Surface**: Input validation gaps, injection vectors, auth assumptions.
  - **Spec Gaps**: Implemented behavior not covered by the spec.
- "Output a numbered list of concrete flaws with file, line, and proposed fix."

### Phase 4 — Feedback Integration Loop
Process the Adversary's feedback:
- **Spec-level flaws** → Update `spec.md`, re-run spec review.
- **Test-level flaws** → Fix/add tests, verify they fail, then fix implementation.
- **Implementation-level flaws** → Refactor, ensure all tests still pass.
- **New edge cases** → Add to spec, write failing tests, implement fixes.

Repeat Phase 3-4 loop until the Adversary's critique contains only nitpicks or NO FLAWS FOUND.

## Phase 5 — Formal Hardening

### Step 5a: Property-Based Testing Verification
Run the property-based tests with high iteration counts. Report results.

### Step 5b: Mutation Testing (if tooling available)
If mutation testing tools are available for the chosen language, run them. Report mutation kill rate. If mutations survive, add tests to cover the gaps.

### Step 5c: Static Analysis
Run language-appropriate static analysis / linting. Fix any findings.

### Step 5d: Purity Boundary Audit
Verify the purity boundaries from Phase 1b are intact. Flag any side effects that crept into the pure core during development.

Write results to `review/verification-report.md`.

## Phase 6 — Convergence

Write `review/convergence-report.md` with a summary table:

| Dimension | Status | Evidence |
|-----------|--------|----------|
| Spec | Converged / Not Converged | Adversary found no spec-level flaws |
| Tests | Converged / Not Converged | All tests pass, mutation kill rate, no tautological tests |
| Implementation | Converged / Not Converged | Adversary forced to invent problems |
| Verification | Converged / Not Converged | Property tests pass, static analysis clean, purity intact |

If all four dimensions show "Converged", the software is **Zero-Slop**.

Present the convergence report to the user.

## Traceability
Every artifact must link back through the VSDD Contract Chain:
```
Spec Requirement → Verification Property → Test Case → Implementation → Adversarial Review
```

At any point, it must be possible to ask "Why does this line of code exist?" and trace it to a spec requirement.

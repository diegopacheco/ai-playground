Verified Spec-Driven Development (VSDD)
The Fusion: VDD × TDD × SDD for AI-Native Engineering
Overview
Verified Spec-Driven Development (VSDD) is a unified software engineering methodology that fuses three proven paradigms into a single AI-orchestrated pipeline:

Spec-Driven Development (SDD): Define the contract before writing a single line of implementation. Specs are the source of truth.
Test-Driven Development (TDD): Tests are written before code. Red → Green → Refactor. No code exists without a failing test that demanded it.
Verification-Driven Development (VDD): Subject all surviving code to adversarial refinement until a hyper-critical reviewer is forced to hallucinate flaws.
VSDD treats these not as competing philosophies but as sequential gates in a single pipeline. Specs define what. Tests enforce how. Adversarial verification ensures nothing was missed. AI models orchestrate every phase, with the human developer serving as the strategic decision-maker and final authority.

I. The VSDD Toolchain
Role	Entity	Function
The Architect	Human Developer	Strategic vision, domain expertise, acceptance authority. Signs off on specs, arbitrates disputes between Builder and Adversary.
The Builder	Claude (or similar)	Spec authorship, test generation, code implementation, and refactoring. Operates under strict TDD constraints.
The Tracker	Chainlink	Hierarchical issue decomposition — Epics → Issues → Sub-issues ("beads"). Every spec, test, and implementation maps to a bead.
The Adversary	Sarcasmotron (Gemini Gem or equivalent)	Hyper-critical reviewer with zero patience. Reviews specs, tests, and implementation. Fresh context on every pass.
II. The VSDD Pipeline
Phase 1 — Spec Crystallization
Nothing gets built until the contract is airtight — and the architecture is verification-ready by design.

The human developer describes the feature intent to the Builder. The Builder then produces a formal specification document for each unit of work. Critically, this phase doesn't just define what the software does — it defines what must be provable about it and structures the architecture accordingly.

Step 1a: Behavioral Specification

The Builder produces the functional contract:

Behavioral Contract: What the module/function/endpoint must do, expressed as preconditions, postconditions, and invariants.
Interface Definition: Input types, output types, error types. No ambiguity. If it's an API, this is the OpenAPI/GraphQL schema. If it's a module, this is the type signature and doc contract.
Edge Case Catalog: Explicitly enumerated boundary conditions, degenerate inputs, and failure modes. The Builder is prompted to be exhaustive here — "What happens when the input is null? Empty? Maximum size? Negative? Unicode? Concurrent?"
Non-Functional Requirements: Performance bounds, memory constraints, security considerations baked into the spec itself.
Step 1b: Verification Architecture

Before any implementation design is finalized, the Builder produces a Verification Strategy that answers: "What properties of this system must be mathematically provable, and what architectural constraints does that impose?"

This includes:

Provable Properties Catalog: Which invariants, safety properties, and correctness guarantees must be formally verified — not just tested? Examples: "This state machine can never reach an invalid state." "This arithmetic can never overflow." "This parser always terminates." "This access control check is never bypassed." The Builder distinguishes between properties that should be proven (critical path, security boundaries, financial calculations) and properties where test coverage is sufficient (UI formatting, logging, non-critical defaults).
Purity Boundary Map: A clear architectural separation between the deterministic, side-effect-free core (where formal verification can operate) and the effectful shell (I/O, network, database, user interaction). This is the most consequential design decision in VSDD — it dictates module boundaries, dependency direction, and how state flows through the system. The pure core must be designed so that verification tools can reason about it without mocking the entire universe.
Verification Tooling Selection: Based on the language and the properties to be proven, the Builder selects the appropriate formal verification stack (Kani for Rust, CBMC for C/C++, Dafny, TLA+ for distributed systems, etc.) and identifies any constraints these tools impose on code structure. This happens now, not after the code is written, because tool constraints are architectural constraints.
Property Specifications: Where possible, the Builder drafts the actual formal property definitions (e.g., Kani proof harnesses, Dafny contracts, TLA+ invariants) alongside the behavioral spec. These aren't implementation — they're the formal expression of what the spec already says in natural language. They serve as a second, mathematically precise encoding of the requirements.
Why this must happen in Phase 1: If the system is designed with side effects woven through the core logic, no amount of Phase 5 heroics will make it verifiable. A function that reads from a database, performs a calculation, and writes to a log in one block cannot be formally verified without mocking infrastructure that the verifier may not support. But a function that takes data in, returns a result, and lets the caller handle persistence — that's a function a model checker can reason about. This boundary must be drawn at the spec level because it fundamentally shapes the module decomposition, the dependency graph, and the testing strategy that follows.

Step 1c: Spec Review Gate

The complete spec — behavioral contracts and verification architecture — is reviewed by both the human and the Adversary before any tests are written. Sarcasmotron tears into the spec looking for:

Ambiguous language that could be interpreted multiple ways
Missing edge cases
Implicit assumptions that aren't stated
Contradictions between different parts of the spec
Properties claimed as "testable only" that should be provable (the Adversary pushes back on lazy verification boundaries)
Purity boundary violations — logic marked as "pure core" that actually depends on external state
Verification tool mismatches — properties the selected tooling can't actually prove
The spec is iterated until the Adversary can't find legitimate holes in either the behavioral contract or the verification strategy.

Chainlink Integration: Each spec maps to a Chainlink Issue. Sub-issues are generated for each behavioral contract item, edge case, non-functional requirement, and each formally provable property. The provable properties get their own bead chain so their status is tracked independently from test coverage.

Phase 2 — Test-First Implementation (The TDD Core)
Red → Green → Refactor, enforced by AI.

With an airtight spec in hand, the Builder now writes tests — and only tests. No implementation code yet.

Step 2a: Test Suite Generation

The Builder translates the spec directly into executable tests:

Unit Tests: One or more tests per behavioral contract item. Every postcondition becomes an assertion. Every precondition violation becomes a test that expects a specific error.
Edge Case Tests: Every item in the Edge Case Catalog becomes a test. These are the tests that catch the bugs that "never happen in production" (until they do).
Integration Tests: Tests that verify the module works correctly within the larger system context defined in the spec.
Property-Based Tests: Where applicable, the Builder generates property-based tests (e.g., using Hypothesis, fast-check, or proptest) that assert invariants hold across randomized inputs.
The Red Gate: All tests must fail before any implementation begins. If a test passes without implementation, the test is suspect — it's either testing the wrong thing or the spec was wrong. The Builder flags this for human review.

Step 2b: Minimal Implementation

The Builder writes the minimum code necessary to make each test pass, one at a time. This is classic TDD discipline:

Pick the next failing test.
Write the smallest implementation that makes it pass.
Run the full suite — nothing else should break.
Repeat.
Step 2c: Refactor

After all tests are green, the Builder refactors for clarity, performance, and adherence to the non-functional requirements in the spec. The test suite acts as the safety net — if refactoring breaks something, the tests catch it immediately.

Human Checkpoint: The developer reviews the test suite and implementation for alignment with the "spirit" of the spec. AI can miss intent even when it nails the letter of the contract.

Phase 3 — Adversarial Refinement (The VDD Roast)
The code survived testing. Now it faces the gauntlet.

The verified, test-passing codebase — along with the spec and test suite — is presented to Sarcasmotron in a fresh context window.

What the Adversary reviews:

Spec Fidelity: Does the implementation actually satisfy the spec, or did the tests inadvertently encode a misunderstanding?
Test Quality: Are the tests actually testing what they claim? Are there tests that would pass even if the implementation were subtly wrong? (Tautological tests, tests that mock too aggressively, tests that assert on implementation details rather than behavior.)
Code Quality: The classic VDD roast — placeholder comments, generic error handling, inefficient patterns, hidden coupling, missing resource cleanup, race conditions.
Security Surface: Input validation gaps, injection vectors, authentication/authorization assumptions.
Spec Gaps Revealed by Implementation: Sometimes writing the code reveals that the spec was incomplete. The Adversary looks for implemented behavior that isn't covered by the spec.
Negative Prompting: Sarcasmotron is prompted for zero tolerance. No "overall this looks good, but..." preamble. Every piece of feedback is a concrete flaw with a specific location and a proposed fix or question.

Context Reset: Fresh context window on every adversarial pass. No relationship drift. No accumulated goodwill.

Phase 4 — Feedback Integration Loop
The Adversary's critique feeds back through the entire pipeline:

Spec-level flaws → Return to Phase 1. Update the spec, re-review.
Test-level flaws → Return to Phase 2a. Fix or add tests, verify they fail against the current implementation (or a deliberately broken version), then fix implementation if needed.
Implementation-level flaws → Return to Phase 2c. Refactor, ensure all tests still pass.
New edge cases discovered → Add to spec's Edge Case Catalog, write new failing tests, implement fixes.
This loop continues until convergence (see Phase 6).

Phase 5 — Formal Hardening (Executing the Verification Plan)
The verification architecture designed in Phase 1b is now executed against the battle-tested implementation. Because the codebase was architected from the start with a pure core and clear purity boundaries, formal verification tools can operate on it without heroic refactoring.

Proof Execution: The property specifications drafted in Phase 1b (Kani harnesses, Dafny contracts, TLA+ invariants, etc.) are run against the implementation. Because the architecture was designed for verifiability, these proofs should engage cleanly with the pure core. Failures here indicate either implementation bugs or spec properties that need refinement — both feed back through Phase 4.
Fuzz Testing: Structured fuzzing (AFL++, libFuzzer, cargo-fuzz) is layered on top of property-based tests to find inputs that no human or AI anticipated. The deterministic core is an ideal fuzz target because it has no environmental dependencies to mock.
Security Hardening: Suites like Wycheproof (cryptographic edge cases) and Semgrep (static analysis) are run as CI/CD gates.
Mutation Testing: Tools like mutmut or Stryker mutate the code to verify the test suite actually catches real bugs. If a mutation survives, the test suite has a gap.
Purity Boundary Audit: A final check that the purity boundaries defined in Phase 1b have been respected throughout implementation. Any side effects that crept into the pure core during development are flagged and refactored out.
All formal verification and fuzzing results feed back into Phase 4 if issues are found.

Phase 6 — Convergence (The Exit Signal)
VSDD inherits VDD's hallucination-based termination, extended across all three dimensions:

Dimension	Convergence Signal
Spec	The Adversary's spec critiques are nitpicks about wording, not about missing behavior, ambiguity, or verification gaps.
Tests	The Adversary can't identify a meaningful untested scenario. Mutation testing confirms high kill rate.
Implementation	The Adversary is forced to invent problems that don't exist in the code.
Verification	All properties from the Phase 1b catalog pass formal proof. Fuzzers find nothing. Purity boundaries are intact.
Maximum Viable Refinement is reached when all four dimensions have converged. The software is considered Zero-Slop — every line of code traces to a spec requirement, is covered by a test, has survived adversarial scrutiny, and the critical path is formally proven.

III. The VSDD Contract Chain
One of VSDD's defining properties is full traceability. Every artifact links back:

Spec Requirement → Verification Property → Chainlink Bead → Test Case → Implementation → Adversarial Review → Formal Proof
At any point, you can ask: "Why does this line of code exist?" and trace it all the way back to a specific spec requirement, through the verification property it satisfies, the test that demanded it, the adversarial review that hardened it, and the formal proof that guarantees it. Equally, you can ask "Why is this module structured as a pure function?" and trace that decision back to the Purity Boundary Map in Phase 1b.

IV. Core Principles of VSDD
Spec Supremacy: The spec is the highest authority below the human developer. Tests serve the spec. Code serves the tests. Nothing exists without a reason traced to the spec.

Verification-First Architecture: The need for formal provability shapes the design, not the other way around. Pure core, effectful shell. If you can't verify it, you architected it wrong — and you find that out in Phase 1, not Phase 5.

Red Before Green: No implementation code is written until a failing test demands it. AI models are explicitly constrained to follow TDD discipline — no "let me just write the whole thing and add tests after."

Anti-Slop Bias: The first "correct" version is assumed to contain hidden debt. Trust is earned through adversarial survival, not initial appearance.

Forced Negativity: Adversarial pressure bypasses the politeness filters of standard LLM interactions. The Adversary doesn't care about your feelings — it cares about your invariants.

Linear Accountability: Chainlink beads ensure every spec item, test, and line of code has a corresponding tracked unit of work. Nothing slips through the cracks.

Entropy Resistance: Context resets on every adversarial pass prevent the natural degradation of long-running AI conversations.

Four-Dimensional Convergence: The system isn't done until specs, tests, implementation, and formal proofs have all independently survived adversarial review.

V. AI Orchestration Notes
VSDD is explicitly designed for multi-model AI workflows:

The Builder benefits from large context windows and strong code generation (Claude, GPT-4, etc.). It needs to hold the full spec, test suite, and implementation simultaneously.
The Adversary benefits from a different model or configuration to avoid shared blind spots. Using a different model family (e.g., Gemini as Adversary when Claude is Builder) introduces genuine cognitive diversity.
The Human is not a bottleneck — they're the strategic layer. They approve specs, resolve disputes, and make judgment calls that AI can't. The human's role is elevated, not diminished, by the AI orchestration.
Prompt Engineering for TDD Discipline: The Builder must be explicitly instructed: "You are operating under strict TDD. Write tests FIRST. Do NOT write implementation code until I confirm all tests fail. When implementing, write the MINIMUM code to pass each test." Without this constraint, AI models will naturally try to write implementation and tests simultaneously.

VI. When to Use VSDD
VSDD is high-ceremony by design. It's worth the overhead when:

Correctness is non-negotiable (financial systems, medical software, infrastructure)
The codebase will be maintained long-term and must resist entropy
Multiple AI models are available and the team wants maximum quality extraction
Security is a primary concern, not an afterthought
The project complexity justifies formal spec work
For rapid prototyping or throwaway scripts, use the parts that make sense — TDD discipline and a quick adversarial pass can still catch a lot of slop even without the full ceremony.

"VSDD doesn't just generate code — it generates code that can prove why it exists, demonstrate that it works, and survive an adversary that wants it dead."

by https://gist.github.com/dollspace-gay/d8d3bc3ecf4188df049d7a4726bb2a00
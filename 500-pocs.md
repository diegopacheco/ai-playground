# 500+ POCs: What did I learned?

Two years. Summer 2024 (300 POCS) to summer 2026. 500+ POCs, 170+ agents, 20+ Claude skills, MCP servers, CV games, RL agents, full-stack clones, local models, and one auction house where AI agents bid on a haunted rubber duck.

Here is the thing nobody tells you about building 500 things: the code is the least valuable output. The code rots in weeks — frameworks rename packages, wheels drop modules, endpoints change defaults. What compounds is something else entirely, and it compounds along exactly four dimensions:

```
🧠 Skill & Judgment   —> Getting Better
⚡ Speed & Delivery    -> Getting Faster
🔬 Culture & Method   —> How you work
❤️ Motivation         —> Why you keep going
```

Eight lessons. Every one backed by code sitting in this repo right now.

---

## 🧠 Dimension 1: Skill & Judgment — getting better

### Lesson 1: Learn New Features — the tool you think you know is 20% of the tool that exists

You do not learn a tool by reading its docs. You learn it when a POC corners you into a capability you did not know existed. I "knew" Playwright for years — then a skill forced me to discover it records video of a browser session, and suddenly bug reports became `.mp4` files of the bug actually happening. Nobody assigned me that. No sprint ticket says "discover that your test runner is also a camera." Features hide until a real problem drags them into the light, and POCs manufacture those problems on demand.

**Proof from the repo:**
- [`pocs/agent-skill-bug-video-recording`](pocs/agent-skill-bug-video-recording) — Playwright doesn't just assert, it *films*: a skill that hunts React bugs and records an optimized video of each one, with a sample app carrying three planted bugs so the whole pipeline is verifiable.
- [`pocs/clihub-fun`](pocs/clihub-fun) — discovered you can compile *any* MCP server into a standalone CLI binary, OAuth included. MCP is not just a protocol for editors; it is a code-generation target.
- [`pocs/cc-token-bar`](pocs/cc-token-bar) — discovered that Claude Code hooks are secretly a telemetry feed: a native macOS menu-bar app showing live token spend, cost, cache-hit ratio, and tools ranked by dollars — fed entirely by hooks Claude Code already calls. The POC started here, then outgrew the playground into [its own repo](https://github.com/diegopacheco/cc-token-bar) — the surest sign a "small experiment" found something real.

### Lesson 2: Acquire Taste — see what's good, bad, works, or doesn't

Taste is not an opinion; it is pattern-matching trained on your own scar tissue, and there is no shortcut. One POC teaches a tool. Ten teach a category. One hundred teach failure patterns. Five hundred teach when *not* to use AI at all — which is the most valuable judgment in the whole set. The non-obvious part: taste comes from running the **same task** through different tools and watching them fail differently. A leaderboard tells you a score; identical tasks across agents tell you *character*.

**Proof from the repo:**
- The Rubik's rubric gauntlet: [`pocs/cc-fable-5-rubiks-cube-poc`](pocs/cc-fable-5-rubiks-cube-poc), [`pocs/codex-5.5-medium-rubrik`](pocs/codex-5.5-medium-rubrik), [`pocs/gemini-3.1-pro-rubrik`](pocs/gemini-3.1-pro-rubrik) — same challenge, three agents, three completely different failure signatures.
- The head-to-head arenas: [`pocs/connect-four-agent-vs-agent`](pocs/connect-four-agent-vs-agent) and [`pocs/agent-werewolf`](pocs/agent-werewolf) — Claude, Gemini, Copilot, and Codex playing *each other* live: Connect Four for tactics, Werewolf for deception — one agent lies while the rest hunt it, and each model gets a deception score. A leaderboard gives you a number; watching a model bluff badly gives you taste.
- [`mcp/llm-judges`](mcp/llm-judges) — taste, industrialized: an MCP server that fans content out to Claude, Codex, Copilot, and Gemini as parallel judges and aggregates PASS/FAIL/SPLIT. When your judges split, you've found exactly where taste lives.

---

## ⚡ Dimension 2: Speed & Delivery — getting faster

### Lesson 3: Emergency → Repetition → Pattern — build the same thing three times and the abstraction introduces itself

Here is the revelation: you cannot design a good abstraction from one data point. The first time you build a datastore console, it is a Redis console. The second time, a SQL console. By the third, your hands are bored — and boredom is the signal. Every datastore console is the same machine: *connect → introspect → query → render a table*. That is when the **generic console** (redis/cassandra/mysql, one shell) becomes obvious — not designed, *discovered*. Repetition is not waste. Repetition is how patterns confess.

**Proof from the repo:**
- [`pocs/redis-fs`](pocs/redis-fs) — a full virtual filesystem shell over Redis in Rust: `ls`, `cat`, `cp`, even executing bash scripts stored in Redis keys. The first "console."
- [`pocs/agent-skill-data-dict`](pocs/agent-skill-data-dict) — discovers a schema from Liquibase, raw SQL, *and* JPA entities, merges them, and ships an in-browser SQL query console. The second "console" — and the introspection pattern crystallizes.
- The generator trilogy: [`agent-runbook`](pocs/agent-runbook), [`agent-skill-infra-automation-generator`](pocs/agent-skill-infra-automation-generator), [`agent-bruno-skill`](pocs/agent-bruno-skill) — one generates operational runbooks with real ports and real failure modes, one generates production infra files, one generates a complete Bruno API collection from detected endpoints. Three different outputs, and by the third the machine is undeniable: *scan the codebase → build a model of the stack → emit the artifact*. Nobody designed that pipeline; three repetitions confessed it.

### Lesson 4: Discovery — 4 years could be done in 4 weeks (100% delivery speed-up, 0% discovery speed-up)

The most uncomfortable insight in this entire repo. Work that took 4 years at a past job could be *rebuilt* today in 4 weeks — the typing, the wiring, the tests, all of it. But here is the trap everyone falls into: AI gave delivery a 100% speed-up and discovery a **0%** speed-up. The 4 years were never about typing. They were about *finding out what to build* — the requirements that only production traffic, angry users, and 3 a.m. pages can reveal. AI compresses construction, not comprehension. Which means discovery is now the entire bottleneck — and the only way to speed it up is to run more experiments, faster, against reality.

**Proof from the repo:**
- The twitter-clone assembly line — [`adwf-twitter-like`](pocs/adwf-twitter-like), [`adwf-twitter-like-codex`](pocs/adwf-twitter-like-codex), [`v3`](pocs/adwf-twitter-like-opus-4-6-v3-skill), [`v4`](pocs/adwf-twitter-like-opus-4-6-v4-skill), [`v5`](pocs/adwf-twitter-like-opus-4-6-v5-skill), and finally [the v5 skill pointed at a brand-new product](pocs/adwf-memory-game-opus-4.6-v5-final) — six builds of the same full-stack app (Rust + React, 106 tests, 85 Playwright e2e), each one driving the next version of the ADWF skill. The first build took 3 full runs and an entire subscription's worth of tokens just to get the e2e suite green. Then the skill started evolving, and each version ships a `mistakes.md` — and *what kind* of mistake it logs climbs a level every version: v3's mistakes are wiring (schema types not matching the backend, crate feature flags); v4's mistakes are contracts (frontend and backend disagreeing on response shapes, wrong status codes, a hardcoded JWT secret caught in review); by v5 the pipeline — design doc → build → four test suites → code + security review → changelog — runs unchanged on a completely different product. Delivery collapsed with every version. The lessons did not: they only exist because the versions can be diffed, and no speed-up can skip that comparison.
- [`pocs/autobench-skill-poc`](pocs/autobench-skill-poc) — a skill that generates a naive baseline, then runs optimization waves against a real benchmark: 411.7ms → 151.3ms, 2.72x faster. Generating each wave's code is instant — that is the 100% delivery speed-up. Knowing whether a wave actually helped requires running it and honestly recording when it made things *worse* — that is the 0%: reality still has to vote on every wave.
- [`pocs/agent-learner-prompt`](pocs/agent-learner-prompt) — the "Lisa loop": an attempt to automate discovery itself. Each cycle generates code, reviews it, extracts learnings and mistakes, and rewrites its own prompt — five LLM calls per cycle, tokens burning like crazy. The README's own verdict: "I'm not happy with the results but it works." Discovery refused to be automated, and that honest failure is the cleanest measurement of the 0% in this whole repo.

---

## 🔬 Dimension 3: Culture & Method — how you work

### Lesson 5: Foster Experimentation Culture — experiments lead to discovery; the more, the better

Since discovery is the bottleneck (Lesson 4), experiment *volume* is the only lever left. Most organizations optimize for experiment *success* — which is backwards, because a POC that cleanly kills a bad idea is a successful POC. The culture shift: make experiments so cheap, so reproducible, so disposable that "let's find out" beats "let's schedule a meeting about it." A run script, a stop script, a test script — boring infrastructure is what makes ambitious experiments possible. And public experiments compound: visible work has to be runnable and explainable, which is a quality mechanism, not marketing.

**Proof from the repo:**
- 536 directories in [`pocs/`](pocs) alone — from [`gelu`](pocs/gelu) (one activation function) to [`k8s-sre-agent-operator`](pocs/k8s-sre-agent-operator) (a Kubernetes operator running an SRE agent). No experiment too small, none too weird.
- The sklearn atlas — ~50 POCs, [`anomaly-detections-IsolationForest`](pocs/sklearn-anomaly-detections-IsolationForest) through [`dimentionality-reduction-UMAP`](pocs/sklearn-dimentionality-reduction-UMAP) — one concept per folder, each independently runnable. Classical ML did not die; it got catalogued.
- 20+ `agent-skill-*` POCs — [`branch-tombs`](pocs/agent-skill-branch-tombs) (a graveyard website for your stale branches), [`bus-factor`](pocs/agent-skill-bus-factor) (git-blame knowledge-risk heatmap), [`claudemd-bs`](pocs/agent-skill-claudemd-bs) (a BS-o-meter for your own agent instructions). Once experimenting is cheap enough, you even experiment on your experiments.

### Lesson 6: Re-wire Risk Perception — more is possible than you think, and "how fast can you say NO?" is a superpower

Two rewirings happen at once. First: your ambition ceiling was miscalibrated — a Kubernetes operator, a Claude Code clone in Rust, a compiled-CLI-from-any-MCP-server all *sound* like team-sized projects until you build one solo in a weekend, and after that your risk perception never recovers. Second, the anti-fragile inversion: as agents get more powerful, the elite skill flips from "how fast can you build?" to "how fast can you say NO?" — permission boundaries, blast-radius design, loud failure. The builders who thrive are not the ones who trust the agent most; they are the ones who made distrust *cheap*.

**Proof from the repo:**
- [`pocs/claude-code-like-rust`](pocs/claude-code-like-rust) — a coding agent CLI built from scratch in Rust. The kind of project you assume needs a company, until you have one in a folder.
- [`pocs/cc-permissions-heatmap`](pocs/cc-permissions-heatmap) + [`pocs/agent-observability`](pocs/agent-observability) — engineering the NO starts with seeing it: one replays every recorded tool call from your transcripts against your permission rules and heatmaps exactly where the agent hits the guardrails — including the calls that were *actually* rejected, ground truth, not modeled; the other traces every agent decision as OpenTelemetry spans into Jaeger. You cannot tune a blast radius you cannot see.
- [`pocs/agent-safehouse-poc`](pocs/agent-safehouse-poc) and [`pocs/leak-detector-skill`](pocs/leak-detector-skill) — containment and leaked-secret scanning as first-class POCs. Security added later feels bolted on because it *is*; here saying NO is designed in before the agent has anything to say yes to.

---

## ❤️ Dimension 4: Motivation — why you keep going

### Lesson 7: Feed Your Passion — learn by doing, spike curiosity, and try; it's fun

Nobody sustains 500 POCs on discipline. Discipline gets you to POC 30; *delight* gets you to 500. The trick is that fun projects are secretly the hardest technical tests — a CV game must react every frame, a music toy must survive real-time audio, a platform shooter needs collision, spawning, and state machines with zero libraries. Weak abstractions that survive a request-response POC collapse instantly under continuous pressure. So the playful POCs are doing double duty: they keep you going, *and* they stress-test harder than any enterprise CRUD ever will.

**Proof from the repo:**
- [`pocs/ai-rpg`](pocs/ai-rpg) — an LLM as Dungeon Master: a text RPG where the world does not exist until you walk into it, generated in real time and streamed token by token over SSE, with every campaign persisted. A toy on the surface; underneath, a stateful real-time streaming system that would pass for a serious product architecture.
- The air-game series — [`air-goalkeeper-game-cv`](pocs/air-goalkeeper-game-cv), [`air-shooting-game-cv`](pocs/air-shooting-game-cv), [`air-fishing`](pocs/air-fishing), [`statue-game-cv`](pocs/statue-game-cv) — your webcam is the controller. MediaPipe hand-tracking under frame-by-frame pressure, disguised as recess.
- [`pocs/agents-auction-hourse`](pocs/agents-auction-hourse) + [`pocs/agent-debate-club`](pocs/agent-debate-club) — AI agents bidding real budgets on a Haunted Rubber Duck and a Time-Travel Microwave; AI agents debating live, each in its own text-to-speech voice, while a judge agent calls the winner. Nobody needed either. That is exactly why they got built — and the four-agent CLI orchestration and SSE streaming behind both got battle-tested for free.

### Lesson 8: Motion Is the Strategy to Build Clarity — you cannot think your way to insight you can only build your way there

The deepest lesson, and the inversion of how planning is supposed to work: clarity is not a *prerequisite* for motion — clarity is a *product* of motion. Every insight in this document is downstream of building, not upstream of it. You do not know what question a POC answers until you are halfway through it; the confusion you feel before starting is not a signal to plan more, it is the raw material the building converts into understanding. Stuck? Confused? Overwhelmed by options? The move is always the same: build the smallest thing that touches reality, and let reality vote.

**Proof from the repo:**
- [`pocs/code-city-viz`](pocs/code-city-viz) — point it at any GitHub repo and it builds a 3D city: every file is a building, lines of code and commit counts set the height, and little people walk in and out of the buildings that change the most. You could stare at `git log` for a week and never *see* the hotspots; one build turned them into a skyline. Clarity was not deduced — it was rendered.
- The Claude Code excavation — [`claude-code-teams`](pocs/claude-code-teams), [`claude-code-loop`](pocs/claude-code-loop), [`claude-context-manager`](pocs/claude-context-manager), [`claude-dispatch`](pocs/claude-dispatch), [`cc-auto-mode-poc`](pocs/cc-auto-mode-poc) — nobody hands you a mental model of how agents orchestrate; a dozen small motions triangulated one.
- [`pocs/12-rules-template-poc`](pocs/12-rules-template-poc) — the working rules that govern this very repo's agents were not designed in a document first. They were extracted from hundreds of sessions of motion, then written down. Method is fossilized motion.

---

## The Spiral

These four dimensions are not a list — they are a flywheel:

**Motion** (❤️) produces experiments. Experiments produce **patterns** (⚡). Patterns produce **taste** (🧠). Taste produces better **method** (🔬). Method makes motion cheaper — and around it goes, each loop faster than the last.

The 500 POCs are not the achievement. They are the *exhaust* of the flywheel. The achievement is the flywheel itself: a person who can now learn any tool by cornering it, spot a pattern by the third repetition, tell good from bad on sight, say NO in milliseconds, and convert confusion into clarity by building.

You do not need 500 POCs to start the flywheel.

You need one. Today. Small enough to finish, real enough to break.

Then another.

That's the whole strategy.

> **Go experiment. Push the frontier — that's how you learn what's possible. Otherwise you just keep your limitations.**

---

## The Opposite

Every lesson has a shadow. This is the same eight, inverted — the person who runs zero POCs, or almost none.

### 🧠 Skill & Judgment

**L1 — Tools stay 20%.** You know the tool exactly as deep as the last ticket required. Playwright is a test runner forever, because nothing ever cornered you into finding the camera. Your depth is capped by your assignments — and assignments are written by people optimizing for delivery, not for your learning.

**L2 — You can't tell plausible from correct.** Taste is the only defense against output that *looks* right, and AI produces looks-right by construction — that's the whole trick. Without scar tissue you have no independent signal, so you fall back on proxies: does it compile, does it match the style guide, does it sound confident, did someone senior nod. All of which the wrong answer passes too. You become a rubber stamp with a job title, approving code you're not equipped to doubt — and the more capable the agent gets, the more expensive that gap becomes, because the errors that survive are the subtle ones.

### ⚡ Speed & Delivery

**L3 — One project's accident becomes your doctrine.** With a single data point you can't tell what was essential from what was circumstance, so the whole thing gets promoted to principle: *we always do it this way*. You carry that shape into problems it doesn't fit, and when it hurts, you assume the problem is wrong rather than the pattern. Worse, you defend it — not from ego, but because you genuinely cannot separate "this design" from "design." Nothing exists to compare it to. Ten years of that is not ten years of experience; it's one year, ratified nine times.

**L4 — You optimize the half that was never the bottleneck.** You buy the tools, run the pilot, measure tokens-per-hour and lines-shipped — all construction metrics, because construction is what you can see. Meanwhile discovery gets nothing: same release cadence, same feedback loop, same three-week gap between shipping and finding out. So you get faster at producing things nobody validated, which is not speed, it's volume. And when the ROI doesn't show up, you conclude the AI underdelivered — because you never had a build where reality voted against you and you had to write that down.

### 🔬 Culture & Method

**L5 — Meetings replace experiments.** "Let's find out" costs a quarter, so "let's schedule a discussion" wins every time. Ideas get killed by argument instead of by evidence — which means the *loudest* idea survives, not the true one. And bad ideas die slowly, in production, at scale.

**L6 — Risk perception stays miscalibrated in both directions.** A Kubernetes operator still sounds team-sized, so you don't start — your ambition ceiling is set by rumor. Simultaneously you *under*-fear agents, because you've never seen one hit a guardrail: distrust is expensive, so you trust by default. You are timid about the safe things and reckless about the dangerous ones.

### ❤️ Motivation

**L7 — Curiosity has to file a business case.** Every hour is justified in advance, so the only things you build are things whose value could be argued *before* they existed — which is precisely the set that excludes everything interesting. The weird idea, the "I wonder if," the thing whose payoff you can only see from the far side: all unfundable, all unbuilt. So you learn only when forced, and forced learning is always late and always shallow. Meanwhile the craft becomes something you survive rather than something you want, and you don't notice the loss, because the work still ships. That's the trap — nothing visibly breaks. You just quietly stop being someone who finds things out.

**L8 — You wait for someone else's clarity.** Confusion registers as a defect — evidence you're not ready — so you go looking for the spec, the RFC, the person who already figured it out. When there isn't one, the work stalls, because your only move for "I don't understand this" is to read more. And the planning feels productive: the doc gets longer, the diagram gets nicer, and none of it has touched anything real. That's what makes it dangerous — pure avoidance would at least feel like avoidance. You end up fluent in problems other people have already solved and mute in front of anything new.

---

**And the flywheel never spins.** No motion → no experiments → no patterns → no taste → no method → motion stays expensive → still no motion. That's the real cost: not eight missing skills, but a stalled loop. The gap compounds against you at the same rate it compounds for the person who started.

**They built their limitations; you inherited yours.**

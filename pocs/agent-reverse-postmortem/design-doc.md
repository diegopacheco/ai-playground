# Design Doc — Reverse Postmortem Skill

## 1. Overview

A Claude Code skill that scans an entire codebase and writes the postmortems of incidents that have **not happened yet**. Instead of analyzing a past outage, it predicts the most likely future outages, ranks them by likelihood and blast radius, and writes each one up as a complete incident report in past tense — timeline, root cause traced to real `file:line`, detection gaps, and preventive action items. The team reads the postmortem of the incident before it occurs and fixes the cause first. Output is a single `reverse-postmortem.md`.

## 2. Problem Statement

Postmortems are written after the damage is done. The lessons are real but expensive — they cost an outage to learn. Teams know their systems have latent failure modes, but those failures stay invisible until they fire at 3am. There is no cheap way to surface "the incident you are most likely to have next" before it happens. Manual pre-mortems exist but are unstructured, rarely grounded in actual code, and easy to skip.

This skill inverts the postmortem: it does the forensic narrative work up front, against the current code, so the most probable incidents are documented and preventable before they ever page someone.

## 3. Goals

- Predict the most likely future incidents from the current codebase
- Rank each predicted incident by **likelihood × blast radius**
- Write each as a full postmortem in past tense (as if it already happened)
- Ground every root cause in real `file:line` evidence, not generic advice
- Construct a plausible, code-derived timeline (trigger → propagation → detection → recovery)
- Identify the **detection gap**: would current logging/metrics/alerts even catch it?
- Produce concrete preventive action items with owners-as-placeholders and effort estimates
- Output everything into a single `reverse-postmortem.md`

## 4. Non-Goals

- Real incident response or live monitoring integration (this is static, pre-incident)
- Replacing actual postmortems of real incidents
- Runtime/dynamic analysis — static code reading only
- Modifying source code — read-only scan
- Security threat modeling (use the threat-analyst skill) or operational runbooks (use the runbook-generator skill). This skill is narrative incident *prediction*, not a procedures catalog.

## 5. Architecture

### 5.1 Scan Pipeline

```
Phase 1: Discover      Phase 2: Hypothesize    Phase 3: Rank          Phase 4: Narrate
+----------------+    +------------------+    +-----------------+    +------------------+
| Map components,|--->| Generate failure |--->| Score by        |--->| Write past-tense |
| deps, hot      |    | hypotheses tied  |    | likelihood x    |    | postmortems with |
| paths, configs |    | to real code     |    | blast radius    |    | timeline + fixes |
+----------------+    +------------------+    +-----------------+    +------------------+
```

### 5.2 Phase 1 — Discovery

Build a model of the system the same way a responder would after being paged:

- **Components**: HTTP endpoints, databases, queues, caches, schedulers, external API calls
- **Hot paths**: code on the critical request path vs. background/rare paths
- **Configs**: timeouts, retries, pool sizes, batch sizes, memory limits, replica counts
- **Fragility signals** (the raw material for hypotheses):
  - Unbounded growth (lists/maps/queues without size limits, no pagination)
  - Missing timeouts on network/DB/external calls
  - No retry, or retry without backoff/jitter (retry storms)
  - Missing circuit breakers around external dependencies
  - Resource leaks (unclosed connections, files, goroutines/threads)
  - N+1 queries, missing indexes implied by query shape
  - Single points of failure (one replica, one broker, one region)
  - Recently changed code (high churn = high risk) via `git log`
  - Error paths that swallow exceptions or log-and-continue
  - Time/date handling, off-by-one in pagination, integer overflow on counters

### 5.3 Phase 2 — Hypothesis Generation

For each fragility signal, generate one or more **incident hypotheses**. A hypothesis is a causal chain, not a single defect:

```
{trigger condition} --> {component fails} --> {propagation} --> {user-visible impact}
```

Hypotheses are matched against an incident-pattern catalog, filtered by what was actually detected:

| Pattern | Trigger | Typical Propagation |
|---|---|---|
| Connection pool exhaustion | Traffic spike + slow query | Requests queue, timeouts cascade, 503s |
| Retry storm / thundering herd | Downstream blip + no backoff | Self-inflicted DDoS, downstream stays down |
| Unbounded memory growth | Cache/list without eviction | Gradual OOM, then CrashLoopBackOff |
| Queue backlog explosion | Consumer slower than producer | Lag grows, DLQ overflows, data delay |
| Cache stampede | Hot key expiry under load | All requests hit DB simultaneously, DB CPU spike |
| Disk fill | Logs/temp files without rotation | Writes fail, DB read-only, service wedged |
| Poison message | Malformed event + no DLQ cap | Consumer crash-loops, partition stalls |
| Cascading timeout | Missing per-call timeout | One slow dep hangs entire request thread pool |
| Clock/TTL bug | Token/cert/TTL boundary | Mass auth failure at a specific timestamp |
| Migration lock | Long DDL on hot table | Writes block, app times out mid-deploy |
| Config drift | IaC vs. app expectation mismatch | Service binds wrong port / can't reach dep |

### 5.4 Phase 3 — Ranking

Each hypothesis gets a **Risk Score = Likelihood × Blast Radius**, each scored 1-5.

**Likelihood (1-5)** — how probable, based on:
- Is the trigger condition common (every traffic spike) or rare (leap second)?
- Is the fragile code on a hot path or a cold path?
- Recent churn in the relevant files (`git log` frequency)
- Presence/absence of guards (a missing timeout scores higher than a present-but-weak one)

**Blast Radius (1-5)** — how much breaks:
- Single endpoint vs. whole service vs. multiple downstream services
- Degradation (slower) vs. full outage vs. data loss/corruption
- Recoverable automatically vs. requires manual intervention

| Risk Score | Tier | Meaning |
|---|---|---|
| 20-25 | P0 | Likely and catastrophic — write up first, fix now |
| 12-19 | P1 | Probable and serious |
| 6-11 | P2 | Plausible, contained |
| 1-5 | P3 | Unlikely or low impact |

Postmortems are written and ordered worst-first.

### 5.5 Phase 4 — Narration

For each ranked hypothesis (above a cutoff), write a full postmortem **in past tense**, as if the incident already occurred. The narrative is fiction grounded in fact: the timeline is invented but every causal link points to real code, real config values, and real dependency edges discovered in Phase 1.

## 6. Output Format — reverse-postmortem.md

```
# Reverse Postmortem — {Project Name}

Generated: {date}
Incidents predicted: {N}   |   P0: {x}  P1: {y}  P2: {z}  P3: {w}

## How to read this
These incidents have NOT happened. Each is the postmortem of a likely future
outage, written in advance so the cause can be fixed before it fires. Ordered
worst-first by Likelihood x Blast Radius.

## Risk Summary

| # | Incident | Likelihood | Blast Radius | Risk | Tier |
|---|----------|-----------|--------------|------|------|
{one row per predicted incident, worst first}

---

## INC-{n}: {Incident Title}

**Tier:** P0 | P1 | P2 | P3
**Risk Score:** {L} x {B} = {score}
**Predicted trigger:** {the condition that would set this off}
**Affected components:** {list}

### Summary
{2-3 sentences, past tense: what happened, who was impacted, how long.}

### Timeline (predicted)
- T+0:00  {trigger fires — tie to real code/config}
- T+0:0X  {first symptom — what a dashboard would show}
- T+0:XX  {propagation step}
- T+0:XX  {detection — or "no alert fired; discovered via user reports"}
- T+0:XX  {mitigation}
- T+X:XX  {recovery}

### Root Cause
{The real defect. Points to exact evidence.}
**Evidence:** `{file}:{line}`
```{snippet of the actual fragile code}```
{Why this code produces the failure under the trigger condition.}

### Contributing Factors
- {missing guard, config value, churn, SPOF — each tied to code}

### Detection Gap
{Would current observability catch this before users did?}
- Existing signals near this path: {logs/metrics found, or "none"}
- What's missing: {the alert/metric that would have caught it early}

### Blast Radius Detail
{What breaks, in what order, and the downstream services affected — derived
 from the Phase 1 dependency map.}

### Action Items (prevent this incident)
| # | Action | File / Area | Effort | Prevents |
|---|--------|-------------|--------|----------|
| 1 | {specific fix} | `{file}:{line}` | S/M/L | {which step in the chain} |

### Earliest Intervention Point
{The single cheapest change that breaks the causal chain — the one thing to do
 if the team only does one thing.}

---

{repeat per incident, worst-first}

## Systemic Patterns
{Cross-incident themes: e.g. "5 of 7 incidents share 'no timeout on external
 calls' — a single client wrapper with sane defaults prevents most of them."}

## Appendix
### Dependency Map (detected)
### Files Scanned
### Scan Methodology
```

## 7. Diagrams

### Causal Chain (per incident)
```
[Trigger]        [Fragile Code]        [Propagation]        [User Impact]
traffic   --->   no timeout on    --->  thread pool    --->  503s on all
 spike           payment API call       exhausted            checkout calls
   |                  |                       |                    |
 common          src/pay.go:88          1 slow dep hangs      revenue-affecting
```

### Intervention Map
```
Trigger --X--> Fragile --X--> Propagation --X--> Impact
        ^cheapest cut here breaks the whole chain
```

## 8. Skill Definitions (two skills)

The skill ships as two cooperating commands: one writes the report, the other renders it as a site.

### 8.1 `reverse-post` — the analyst
- **name:** `reverse-post`
- **description:** Scans the codebase, predicts the most likely future incidents, and writes each as a full past-tense postmortem (timeline, root cause at file:line, detection gap, preventive action items) into `reverse-postmortem.md`.
- **allowed-tools:** `[Glob, Grep, Read, Bash, Write]`
- **trigger:** `/reverse-post`
- **output:** `reverse-postmortem.md` in project root
- **invocation:**
  - `/reverse-post` — scan whole repo
  - `/reverse-post src/payments/` — scope to a path
  - `/reverse-post --top 5` — only write the N highest-risk incidents

### 8.2 `reverse-post-site` — the renderer
- **name:** `reverse-post-site`
- **description:** Reads `reverse-postmortem.md` and renders a self-contained static site (`reverse-postmortem-site/index.html`) — risk dashboard, per-incident cards with severity colors, causal-chain visuals, and action-item checklists. No build step, no external dependencies.
- **allowed-tools:** `[Read, Glob, Bash, Write]`
- **trigger:** `/reverse-post-site`
- **input:** `reverse-postmortem.md` (must exist; if missing, instruct the user to run `/reverse-post` first)
- **output:** `reverse-postmortem-site/index.html` (single self-contained file, inline CSS/JS)
- **invocation:**
  - `/reverse-post-site` — render the report in the project root
  - `/reverse-post-site path/to/reverse-postmortem.md` — render a specific report

### 8.3 Pipeline
```
/reverse-post  ->  reverse-postmortem.md  ->  /reverse-post-site  ->  reverse-postmortem-site/index.html
   (analyze)         (machine + human            (render)              (open in browser)
                       readable report)
```
The report is the contract between the two skills. `reverse-post` owns analysis; `reverse-post-site` owns presentation and never re-analyzes code — it only transforms the report.

## 9. Key Design Decisions

### Why write in past tense?
Past-tense narrative forces specificity. "The cache could grow unbounded" is easy to ignore. "At T+47min the pod was OOMKilled and checkout returned 503 for 9 minutes" makes the cost concrete and the fix urgent. The postmortem format is a behavior-change tool, not just a report.

### Why likelihood × blast radius instead of a flat severity?
A catastrophic-but-impossible failure and a trivial-but-constant failure both rank low; the multiplication surfaces the incidents that are both probable and damaging. This is the same prioritization on-call engineers do intuitively — made explicit.

### Why ground every root cause in file:line?
Predictions without evidence are horoscopes. Tying each hypothesis to real fragile code (a missing timeout at `src/pay.go:88`, a churn-heavy migration file) keeps the output falsifiable and actionable, and lets the reader verify the claim.

### Why a detection gap section?
The worst incidents are the ones nobody sees coming because no alert fires. Checking whether existing logs/metrics would catch each predicted failure turns the skill into an observability audit as a side effect.

### Why "earliest intervention point"?
Teams won't do every action item. Naming the single cheapest cut in the causal chain gives a clear, low-effort win per incident and respects limited engineering time.

### Why use git churn as a likelihood signal?
Recently and frequently changed files are statistically more incident-prone. `git log` is free signal already in the repo; ignoring it would discard the best available proxy for "where bugs actually land."

## 10. Suggestions and Improvements

### What makes this skill valuable
- Turns latent risk into a concrete, prioritized, evidence-backed list — before the outage
- Doubles as a detection-gap / observability audit
- Output reads like a real postmortem, so it slots into existing review rituals
- Falsifiable: every prediction points to code the reader can inspect

### Potential enhancements (future)
- Diff mode: re-run after fixes and show which predicted incidents were retired
- Confidence labels per hypothesis (high/medium/low) alongside the risk score
- Ingest real past postmortems to calibrate the pattern catalog to this team's history
- CI gate: fail if a new P0 incident is introduced by a PR

## 11. Critiques and Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Predictions are probabilistic | Some predicted incidents never happen | Rank by risk, label confidence, point to evidence so readers judge |
| Static analysis only | Misses runtime-only triggers (race conditions, real traffic shape) | State clearly; recommend load testing for the top incidents |
| Timeline numbers are invented | Could be mistaken for real data | Header explicitly marks incidents as predicted, not historical |
| Pattern catalog is finite | Novel failure modes won't be predicted | Combine catalog with code-reading; flag low-confidence inferences |
| Likelihood scoring is heuristic | Ranking may be imperfect | Show the scoring inputs so users can re-weight |
| Large monorepos slow to scan | Timeout risk | Support path scoping and `--top N` cutoff; skip vendor/generated code |

## 12. File Structure

```
~/.claude/skills/reverse-post/
  SKILL.md               -- The analyst skill
~/.claude/skills/reverse-post-site/
  SKILL.md               -- The renderer skill
agent-reverse-postmortem/
  design-doc.md          -- This document
  README.md              -- Usage and overview
  install.sh             -- Installs both skills into ~/.claude/skills/
  uninstall.sh           -- Removes both skills
  reverse-post/SKILL.md
  reverse-post-site/SKILL.md
  sample/                -- A sample service with intentional fragility signals
```

# Closed-Loop Auto-Tuning Agent for Resilience4j

## 1. Summary

A control panel that lets a human exercise four Resilience4j patterns under
controllable load, then asks an LLM to propose a better Circuit Breaker
configuration based on the metrics the load produced. The LLM only *advises*;
a deterministic clamp bounds its proposal, and a human applies it with one
click. The dashboard then re-runs the same load against the new config and
shows before/after charts so the operator can judge whether it actually helped.

This is the **human-gated, on-demand** form of auto-tuning: the loop is closed
through a person pressing a button, not autonomously. That choice is what keeps
it safe (see Section 3).

## 2. Goals and Non-Goals

### Goals
- Backend (Java 25, Spring Boot 4.x, Resilience4j) exposing four protected
  endpoints: Circuit Breaker, Retry, Rate Limiter, Bulkhead.
- A controllable flaky downstream so real traffic from the UI forces each
  pattern into action (open circuit, retries, 429s, rejections).
- An LLM-backed advisor that reads live Circuit Breaker metrics + current
  config and proposes a new Circuit Breaker config as structured JSON.
- A deterministic clamp that bounds the proposal to safe ranges so it can never
  disable protection. Clamped fields are surfaced to the operator.
- One-click apply, then a before/after comparison (two overlaid charts + KPI
  delta table) so "is it better?" is answerable.
- `start.sh` / `stop.sh` to bring the whole stack up and down natively.
- `README.md` with screenshots captured via Playwright.

### Non-Goals (and why)
- **Autonomous tuning.** No background loop mutating config on a timer. The
  human is the controller and the freeze switch. This sidesteps oscillation,
  incident amplification, and reward-hacking (Section 3) without building the
  full guardrail stack those would require.
- **Tuning patterns other than Circuit Breaker.** Retry / Rate Limiter /
  Bulkhead are exercisable and observable, but only the Circuit Breaker is
  tunable by the LLM in this POC. The tuning pipeline is structured so adding
  others later is mechanical.
- **Online reinforcement learning.** No exploration in a live reliability layer.
  The LLM reasons from the metrics it is shown; it does not "learn" across runs.
- **Persistence / multi-user.** State lives in-memory in the backend for the
  session. Restart = clean slate.

## 3. Why closing the loop is the hard part (design constraints)

These are the failure modes that shaped the design. Each maps to a concrete
constraint below.

1. **Oscillation / hunting** — a controller that reacts faster than the system
   settles will overcorrect and swing. *Constraint:* tuning is one-shot and
   human-paced, never on a fast timer.
2. **No ground-truth objective → Goodhart** — "minimize breaker openings" is
   optimally solved by disabling the breaker. *Constraint:* the LLM is given
   explicit SLO intent in its prompt, and the clamp makes "disable protection"
   unreachable regardless of what it returns.
3. **Incident amplification** — during a real outage, metrics look bad because
   they *should*; an autotuner that "makes metrics healthy" relaxes protection
   at the worst moment. *Constraint:* no autonomous application; a human reads
   the rationale and decides.
4. **State reset on apply** — R4J config is immutable; applying new values
   replaces the breaker and drops its sliding window. *Constraint:* before/after
   is measured by deliberately re-running identical load against a fresh window,
   so the reset is part of the methodology rather than a hidden transient.

The takeaway baked into the architecture: **the LLM does qualitative reasoning
and proposes; deterministic code enforces bounds; the human gates.**

## 4. Architecture

```
+--------------------------------------------------------------+
|  Browser (React + Vite + TanStack Query)                     |
|                                                              |
|  [Pattern tabs: CB | Retry | RateLimiter | Bulkhead]         |
|  [Scenario: failRate, latency, jitter, rps, count]           |
|  [Run traffic] --real fetch() bursts-->                      |
|  [Live metrics chart]  [Call LLM to self-tune]               |
|  [Tuning panel: current vs proposed vs clamped + rationale]  |
|  [Apply]  [Before/After charts + KPI delta]                  |
+----------------------------|---------------------------------+
                             | HTTP (fetch)
                             v
+--------------------------------------------------------------+
|  Backend (Spring Boot 4 / Java 25)                           |
|                                                              |
|  PatternController     /api/{cb,retry,ratelimiter,bulkhead}  |
|     -> R4J decorators  -> Downstream (flaky simulator)       |
|                                                              |
|  ScenarioController    /api/sim/scenario  (set failRate...)  |
|  MetricsController     /api/metrics/circuitbreaker  (poll)   |
|  ConfigController      GET/POST /api/config/circuitbreaker   |
|  TuneController        POST /api/tune/circuitbreaker         |
|       -> TuningService -> OpenAiClient (JDK HttpClient)      |
|                        -> Clamp (deterministic bounds)       |
+----------------------------|---------------------------------+
                             | HTTPS
                             v
                    OpenAI API (OPENAI_API_KEY)
```

### Data flow for one tuning cycle
1. Operator picks the **Circuit Breaker** tab and a scenario (e.g. failRate
   0.6, latency 300ms). UI POSTs the scenario to the backend.
2. Operator clicks **Run traffic**. The browser fires N real `fetch()` calls at
   `/api/cb/call` at the chosen rate. Each call flows through the R4J Circuit
   Breaker into the flaky `Downstream`. The UI records the per-call outcome and
   polls `/api/metrics/circuitbreaker` to build the **before** time series.
3. Operator clicks **Call LLM to self-tune**. Backend snapshots current config +
   metrics, builds a prompt, calls OpenAI, parses the JSON proposal, runs it
   through the **Clamp**, and returns `{ current, proposed, clamped, rationale }`.
4. UI shows current vs proposed vs clamped in a diff table; clamped fields are
   highlighted ("LLM said 95% → bounded to 70%").
5. Operator clicks **Apply**. Backend rebuilds the Circuit Breaker in the
   registry with the clamped config (fresh sliding window).
6. Operator clicks **Run traffic** again (same scenario). UI records the
   **after** series, overlays both charts, and renders a KPI delta table.

## 5. Backend design

### 5.1 The flaky downstream
A single `Downstream` bean simulates an unreliable dependency. Its behavior is
held in a mutable `Scenario` (failRate 0..1, latencyMs, jitterMs) set via
`POST /api/sim/scenario`. On each call it sleeps `latency ± jitter` and throws
with probability `failRate`. This is what makes UI traffic *actually* trip R4J,
and keeping the scenario fixed across before/after makes the comparison fair.

### 5.2 Protected endpoints
Each pattern wraps `downstream.call()` with its R4J decorator:

| Endpoint                  | Pattern        | Observable effect under load            |
|---------------------------|----------------|-----------------------------------------|
| `POST /api/cb/call`        | CircuitBreaker | state OPEN → calls short-circuited      |
| `POST /api/retry/call`     | Retry          | retried attempts, then success/failure  |
| `POST /api/ratelimiter/call`| RateLimiter   | excess calls rejected (429)             |
| `POST /api/bulkhead/call`  | Bulkhead       | concurrent calls beyond limit rejected  |

Responses carry `{ outcome: SUCCESS|FAILURE|SHORT_CIRCUITED|RATE_LIMITED|REJECTED, latencyMs }`.

### 5.3 Metrics
`GET /api/metrics/circuitbreaker` returns a flat snapshot read from the R4J
registry + breaker metrics:
```
{ state, failureRate, slowCallRate, bufferedCalls, failedCalls,
  successfulCalls, notPermittedCalls, ts }
```
The UI polls this (≈1s) during a traffic run to build the time series. Equivalent
read-only snapshots exist for the other three patterns to drive their live panels.

### 5.4 Config get / apply
- `GET /api/config/circuitbreaker` → current config knobs.
- `POST /api/config/circuitbreaker` → validate, **clamp again** (defense in
  depth), rebuild the breaker in the `CircuitBreakerRegistry`, and swap the
  reference used by `PatternController`. R4J config objects are immutable, so
  "apply" = construct a new `CircuitBreakerConfig` + new breaker. This resets
  the sliding window by design (Section 3.4).

Tunable knobs:
```
failureRateThreshold, slowCallRateThreshold, slowCallDurationThresholdMs,
slidingWindowType (COUNT|TIME), slidingWindowSize, minimumNumberOfCalls,
waitDurationInOpenStateSeconds, permittedNumberOfCallsInHalfOpenState
```

### 5.5 Deterministic clamp (the guardrail)
Pure function applied to every proposal before it can reach the operator, and
again on apply. Hard bounds:

| Knob                                  | Min | Max  |
|---------------------------------------|-----|------|
| failureRateThreshold (%)              | 40  | 70   |
| slowCallRateThreshold (%)             | 50  | 100  |
| slowCallDurationThresholdMs           | 200 | 5000 |
| slidingWindowSize                     | 10  | 200  |
| minimumNumberOfCalls                  | 5   | 100  |
| waitDurationInOpenStateSeconds        | 5   | 60   |
| permittedNumberOfCallsInHalfOpenState | 2   | 20   |

Clamping records, per field, `{ proposed, applied, wasClamped }` so the UI can
show exactly where the LLM was overruled. There is no value the LLM can return
that turns the breaker off.

### 5.6 LLM integration
- Transport: JDK `java.net.http.HttpClient` (no SDK, no extra HTTP library).
- Endpoint: OpenAI Chat Completions, JSON response format (structured output).
- Auth: `OPENAI_API_KEY` from env. Model from `OPENAI_MODEL` (default `gpt-4o`).
- JSON: the Spring-managed Jackson `ObjectMapper` (note: Spring Boot 4 ships
  Jackson 3 under the `tools.jackson` package — inject the bean rather than
  importing a fixed package).
- If `OPENAI_API_KEY` is unset, `/api/tune` returns `503` with a clear message
  so the dashboard degrades honestly instead of faking a result.

**Prompt shape** (system + user):
- *System:* role = Resilience4j tuning advisor; the SLO intent ("serve as many
  requests successfully as possible while still tripping fast when the
  downstream is genuinely unhealthy; do not mask real failures"); the exact hard
  bounds; required strict JSON schema for the answer.
- *User:* current config + the metrics snapshot/series the run produced.
- *Response:* `{ failureRateThreshold, slowCallRateThreshold,
  slowCallDurationThresholdMs, slidingWindowType, slidingWindowSize,
  minimumNumberOfCalls, waitDurationInOpenStateSeconds,
  permittedNumberOfCallsInHalfOpenState, rationale }`.

## 6. Frontend design

### 6.1 Stack
- **Vite + React + TypeScript.**
- **TanStack Query** for all server state (metrics polling, config, tune call).
- **TanStack Router** for the four pattern routes/tabs.
- **Charts: hand-rolled SVG** line-chart component (no charting library), to
  honor the "fewest dependencies" rule. It renders one or two overlaid series
  with axes and a legend — enough for before/after.

### 6.2 Screens / components
- `PatternTabs` — CB | Retry | RateLimiter | Bulkhead.
- `ScenarioControls` — failRate, latencyMs, jitterMs, rps, request count.
- `TrafficRunner` — fires real `fetch()` bursts at the active endpoint at the
  chosen rate; tallies outcomes; owns the metrics polling that builds a series.
- `LiveMetrics` — current state + counters for the active pattern.
- `TuningPanel` (CB tab only) — **Call LLM to self-tune** button; diff table of
  current vs proposed vs clamped with clamped fields highlighted; rationale
  text; **Apply** button.
- `CompareView` — overlaid before/after charts + KPI delta table.

### 6.3 Before/after KPIs
Computed client-side from the two recorded runs:
- Success rate served (%)
- Short-circuited (not-permitted) calls
- Failed calls reaching the downstream
- Observed mean / p95 client latency
- Time the breaker spent OPEN (s)

The KPI table shows deltas; the operator judges "better" against the scenario
(a higher failure threshold can serve more during a brief blip but mask a real
outage — the point is to make the trade-off visible, not to declare a winner).

## 7. Project layout

```
closed-loop-auto-tunning-agent/
  design-doc.md
  README.md
  start.sh
  stop.sh
  test.sh
  backend/
    pom.xml
    src/main/java/com/diegopacheco/autotune/
      Application.java
      downstream/   Downstream, Scenario, CallOutcome
      pattern/      PatternController, R4jConfig (registry beans)
      metrics/      MetricsController
      config/       ConfigController, CircuitBreakerConfigDto, Clamp
      tune/         TuneController, TuningService, OpenAiClient, TuneResult
    src/main/resources/application.yml
    src/test/java/com/diegopacheco/autotune/   (Clamp + tuning + endpoint ITs)
  frontend/
    package.json  vite.config.ts  index.html
    src/
      main.tsx  App.tsx  router.tsx
      api/        query hooks (metrics, config, tune)
      components/ PatternTabs, ScenarioControls, TrafficRunner, LiveMetrics,
                  TuningPanel, CompareView, LineChart
  docs/screenshots/   (Playwright PNGs referenced by README)
```

## 8. Scripts

- **`start.sh`** — start backend (`./mvnw spring-boot:run`) and frontend
  (`npm install` if needed, `npm run dev`) in the background, write PIDs to
  `.run/`, then poll `http://localhost:8080` health and `http://localhost:5173`
  in a loop (max sleep 1) until both answer. No comments, no emojis.
- **`stop.sh`** — read PIDs from `.run/`, kill both, clean up.
- **`test.sh`** — run backend tests (`./mvnw test`) and print the output (used
  for the PR per project conventions).

Ports: backend `8080`, frontend `5173`. Frontend proxies `/api` to `8080` via
Vite so the browser makes same-origin calls.

## 9. Testing strategy

- **Clamp unit tests** — the core safety property: assert every out-of-range
  proposal (including the adversarial "disable protection" one, e.g. failureRate
  100 / wait 0) is forced inside bounds, and in-range values pass untouched.
  This test must fail if the bounds are ever loosened — it encodes *why* the
  clamp exists, not just that it runs.
- **Endpoint integration tests** — drive each pattern with load and assert the
  expected R4J effect (breaker opens past threshold, rate limiter rejects excess,
  bulkhead rejects over-concurrency, retry retries). Per the Spring Boot 4
  classpath shift, ITs use the JDK `HttpClient` + string JSON rather than
  `TestRestTemplate`.
- **TuningService test** — with the OpenAI call stubbed, assert a raw proposal
  is parsed and then clamped before being returned.

## 10. Screenshots (README)

After the stack is up, Playwright drives the dashboard and captures:
1. Dashboard idle (pattern tabs + scenario controls).
2. Mid traffic run with the live metrics chart climbing / breaker OPEN.
3. Tuning panel showing current vs proposed vs clamped + rationale.
4. Before/after comparison charts + KPI delta table.

PNGs land in `docs/screenshots/` and are embedded in `README.md`.

## 11. Versions

- Java 25 (Corretto 25.0.2 verified locally)
- Spring Boot 4.x, Resilience4j (Spring Boot 3+ starter line)
- Node 24 / npm 11, Vite, React, TypeScript, TanStack Query + Router
- OpenAI Chat Completions via `OPENAI_API_KEY` (model via `OPENAI_MODEL`,
  default `gpt-4o`)

## 12. Risks and honest limitations

- **Apply resets the sliding window.** Documented and made part of the
  before/after method; not hidden.
- **The LLM can still be wrong within bounds.** The clamp prevents catastrophe,
  not sub-optimality; the human reading the rationale is the real check.
- **"Better" is scenario-dependent.** The dashboard surfaces the trade-off; it
  does not claim an objective optimum.
- **No autonomy on purpose.** If this were ever made autonomous, it would need
  the rest of the guardrail stack (cooldown/hysteresis, timescale separation,
  shadow→canary promotion, freeze-on-incident, control group) — explicitly out
  of scope here.

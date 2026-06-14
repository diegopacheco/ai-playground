# Design — rerender

## Goal
A `/rerender` agent skill that runs a real re-render benchmark on every React component in a project and renders a Lighthouse-style, light-theme report. Honest numbers only.

## The measurement
The hard part of a re-render tool is being truthful. Static analysis alone ("this isn't wrapped in `memo`") guesses; it does not measure. So the engine measures.

For each component the engine generates a harness that:
- imports the component, resolving **the project's own React** (esbuild bundle with `nodePaths` pointed at the project's `node_modules`),
- mounts it in **jsdom** inside a `<Profiler>`,
- re-renders its parent **40 times** while passing **referentially-stable props**,
- records every `onRender` commit's `actualDuration`.

A re-render that did measurable work despite unchanged props is a **wasted render** — exactly what `React.memo` exists to skip. A memoized component bails out and records zero; an un-memoized one re-renders all 40 times. This is the canonical "wasted render" lens and it falls straight out of the Profiler, no instrumentation of user code required.

### Why a development build
React's `<Profiler>` only reports timings in a development (or special profiling) build, so the harness is bundled with `NODE_ENV=development`. Absolute durations run a little high, but they are real and comparable across components. The report says so plainly.

### Noise floor
A truly bailed-out subtree occasionally produces a sub-microsecond commit reading. Anything under `0.05 ms` is treated as below measurement resolution and not counted as a render — so a memoized component reads a clean zero instead of a flickering "1 of 40". A genuine re-render in this build costs 0.4–7 ms, well above the floor.

### Root component
The component rendered by `createRoot` has no real parent, so the wasted-render churn test does not apply. The engine detects it from the entry file (skipping `StrictMode`/`Fragment` wrappers), measures its mount cost, and exempts it from the churn test.

## Isolation
The harness is bundled to CommonJS and run in a **separate node process** (`jsdom` stays external and resolves via `NODE_PATH`). This keeps the project's React out of the engine's own module state and guarantees a clean exit even if a component starts timers.

## Scoring
Lighthouse-style, 0–100, green ≥90 / orange 50–89 / red <50:
- wasted-render ratio → up to −60,
- average render cost → up to −30,
- mount cost → up to −12,
- static anti-patterns (inline props, unmemoized array work) → small penalties.

Static analysis (memo, hook counts, inline props, array ops, LOC) enriches the per-component cards and feeds the issue list, but the headline numbers are always the measured ones.

## Report
A single self-contained `index.html`: overall score gauge, animated metric counters, a wasted-renders bar chart, a score distribution, an average-render-cost chart, and a sortable/filterable card per component (score ring, tags, hooks, counters, per-render sparkline, and a concrete fix per issue). Data is embedded as JSON; no server or network.

## Sample app
A Vite + React 19 dashboard with six components: four correctly memoized, two with deliberate re-render issues (`ActivityFeed`, `MetricsTable`) that are both un-memoized and do heavy per-render work. It is the skill's own fixture and the screenshots in the README.

## Non-goals
- Not a replacement for the React DevTools Profiler on a running app — it is a CI-friendly, per-component benchmark.
- Components that need required props and cannot be auto-mounted are reported as not-measured, not estimated.
- Projects needing path aliases or exotic loaders surface an esbuild error rather than a partial report.

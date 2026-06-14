---
name: rerender
description: Runs a real re-render benchmark on every React component in a project and renders a Lighthouse-style light-theme performance report. For each component it mounts the real code in jsdom inside a React Profiler, forces a fixed number of parent re-renders with stable props, and counts the renders React could have skipped (wasted renders) plus render durations, then scores each component 0-100. Use when the user runs /rerender or asks to measure, benchmark, or find React re-render / wasted-render / memoization performance problems.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# React Re-render Performance

When invoked, you measure how every React component in a project behaves under parent re-renders and render a light-theme, Lighthouse-style website that scores each component 0-100. The engine bundles a harness with the project's **own React**, mounts each component in **jsdom** inside a `<Profiler>`, re-renders its parent a fixed number of times with **referentially-stable props**, and uses React's `onRender` callback to count the re-renders that did real work despite unchanged props — the wasted renders that `React.memo` would skip — plus mount and per-render durations.

## Global Context
- User request / scope: $ARGUMENTS — empty, or a path to the React project root
- Engine: `scripts/rerender.mjs` (Node; only dependencies are esbuild and jsdom, installed beside the skill)
- Template: `assets/template.html`
- Output: `rerender-report/index.html` and `rerender-report/data.json` in the current directory

## Rules
- Every number is **measured at runtime, never estimated**. If the engine cannot mount a component, it records that component as not-measured rather than inventing numbers.
- Read-only against the project. The skill only writes inside `rerender-report/` (and a `.rerender-tmp/` it deletes when done).
- The engine needs the project's `node_modules` (it mounts the project's real React and components). If they are missing, it cannot run.
- Do not add comments to any command you run.

## Step 1 — Find the project
Decide the target from `$ARGUMENTS`:
- A path to a directory → that is the project root.
- Empty → use the current working directory.

Confirm it is a React project: a `package.json` exists and `react` is installed under `node_modules`. The engine scans `src/` if present, otherwise the project root, for `.jsx/.tsx` (and React-flavored `.js/.ts`) component files, skipping `node_modules`, build output, tests, stories, and the entry file that calls `createRoot`/`hydrateRoot`.

## Step 2 — Make sure dependencies are installed
The harness mounts the project's real React, so its dependencies must resolve:

```bash
test -d <project>/node_modules/react && echo present || echo missing
```

If missing, tell the user the engine needs an install first and offer to run it:

```bash
npm --prefix <project> install
```

## Step 3 — Run the benchmark engine
Invoke the engine by its installed absolute path. It always writes `rerender-report/` into the current working directory, so run it from where you want the report. Pass the project root:

```bash
cd <project> && node "$HOME/.claude/skills/rerender/scripts/rerender.mjs" .
```

The engine discovers the components, statically analyses each (memoization, hooks, inline props, array work in render), then bundles and runs the measurement harness once. It prints a per-component summary to stdout and writes the full report. If it cannot bundle the project (path aliases, exotic loaders) it explains why — relay that instead of producing a partial report.

## Step 4 — Report back
Relay the stdout summary in your own words: the overall score, how many components are clean vs. wasting renders, the total wasted renders out of stressed updates, and the slowest component. Name the components that scored poorly and the one-line fix for each (wrap in `React.memo`, move heavy work into `useMemo`, stop passing inline object/array/function props). Point the user at the generated site:

```
rerender-report/index.html
```

Offer to open it. The page is self-contained — an overall score gauge, animated metric counters, a wasted-renders bar chart, a score distribution, an average-render-cost chart, and a sortable/filterable card per component with its score, metrics, a per-render sparkline, and concrete fixes — no server or network needed.

## What the numbers mean
- **Wasted renders** — under N parent re-renders with unchanged props, the count of re-renders that still did measurable work. A correctly memoized component bails out and scores 0; an un-memoized one re-renders every time. This is the headline re-render lens.
- **Avg / peak ms** — `actualDuration` from React's Profiler. Measured in a React **development profiling build** (required for the Profiler to report timings), so absolute values run a little high but are honest and comparable across components.
- **Score** — 0-100, Lighthouse-style: green ≥90, orange 50-89, red <50. Driven mainly by the wasted-render ratio, plus render cost and static anti-patterns.
- **Root component** — the one rendered by `createRoot` has no real parent, so it is measured for mount cost only and exempt from the wasted-render test.

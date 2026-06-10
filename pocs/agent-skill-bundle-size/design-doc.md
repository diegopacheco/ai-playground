# Design doc — bundle-size attribution skill

## Goal

Read a frontend project and report which import added the most KB to its JavaScript bundle, then render a light-theme analytics website. The headline output is a single ranked truth: the heaviest import and the minified KB it contributed.

## Why esbuild's metafile

The only honest way to say "this import added N KB" is to bundle the real source graph and read what the bundler actually emitted. esbuild produces a `metafile` whose `outputs[].inputs[path].bytesInOutput` is the exact minified contribution of each input module. That makes attribution measured, not modelled. esbuild is a single, fast, zero-config binary — the minimum viable dependency for the job. Source-size heuristics (counting characters in `node_modules`) were rejected because they ignore tree-shaking and minification and would lie.

## Pipeline

```
project root
  -> find entry (index.html script, or src/main.{jsx,tsx,js,ts}, ...)
  -> esbuild.build({ bundle, minify, metafile, write:false, format:esm })
  -> per-input bytesInOutput summed across outputs (JS + CSS)
  -> group by package: node_modules/<pkg> | @scope/pkg | (application code)
  -> totals: minified (sum output bytes), gzip (zlib over the output), raw (sum input bytes)
  -> inject JSON into assets/template.html -> bundle-size-report/index.html
```

## Attribution rules

- A module's package is the last `node_modules/` segment in its path (handles nested and pnpm layouts via `lastIndexOf`). Scoped packages keep `@scope/name`.
- Anything outside `node_modules` is one bucket: `(application code)`.
- Package version is read from `node_modules/<pkg>/package.json` — reported as found, never inferred.
- The headline is the single package with the largest summed `bytesInOutput`. App code can win; if it does, the report says so rather than forcing a vendor answer.
- Gzip is reported only for the whole bundle. Per-package gzip is intentionally omitted: gzip is not additive across modules, so a per-import gzip number would be misleading.

## Robustness choices

- Asset imports (png/svg/woff/…) get `dataurl` loaders and `.js` gets the `jsx` loader so CRA-style and asset-heavy projects bundle without config.
- `process.env.NODE_ENV` and `import.meta.env.*` are defined to production so dev-only branches drop out, matching a real production build.
- Missing `node_modules` and a missing entry both fail loudly with an actionable message; esbuild failures (SCSS/LESS, vite aliases) are surfaced verbatim instead of producing a partial report.

## The report

Self-contained HTML, light theme, no build step, no network:
- Hero ring on the heaviest import + KPI cards (minified, gzip, raw, imports, vendor share, app code).
- Squarified treemap of imports, colored by share (green small -> red large), app code in blue.
- Heaviest-imports bar list and an app-vs-vendor donut.
- Sortable, filterable "Every import" table with per-row micro-bars.
- Module explorer listing the individual files in the bundle.

Data is injected at the `__BUNDLE_DATA__` marker, with `</` escaped so the JSON cannot break out of the `<script>`.

## Packaging

- `skills/bundle-size/` is the installable unit: `SKILL.md` (agent playbook), `scripts/bundlesize.mjs` (engine), `assets/template.html`, `package.json` (esbuild).
- `install.sh` copies it to `~/.claude/skills` (and `~/.codex/skills` when present) and runs `npm install` for esbuild.
- `sample-app/` is a Vite + React dashboard with a deliberate spread of dependency sizes so the report is meaningful.

## Non-goals

- Not a replacement for a project's own bundler stats; it is a fast, portable second opinion on the same source graph.
- No per-route / code-splitting analysis — it attributes one entry's bundle.
- No SCSS/LESS/alias resolution; those projects are reported as esbuild errors.

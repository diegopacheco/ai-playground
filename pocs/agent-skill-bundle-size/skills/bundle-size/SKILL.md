---
name: bundle-size
description: Attributes a frontend project's JavaScript bundle size to the imports that caused it, then renders a light-theme analytics website. Bundles the app with esbuild, reads the metafile, and reports how many minified KB each npm package and source module added. Use when the user runs /bundle-size or asks which import is the heaviest, what is bloating the bundle, or for a bundle-size / import-cost report.
allowed-tools: [Bash, Read, AskUserQuestion]
---

# Bundle Size Attribution

When invoked, you measure where a frontend bundle's bytes come from and render a light-theme website that ranks every import by the minified KB it added. The engine bundles the project with esbuild, reads esbuild's metafile, and attributes each module's `bytesInOutput` back to the npm package (or app file) that brought it in. The headline answer is the single import that added the most KB.

## Global Context
- User request / scope: $ARGUMENTS — empty, a path to the frontend project, or an explicit entry file
- Engine: `scripts/bundlesize.mjs` (Node, only dependency is esbuild, installed beside the skill)
- Template: `assets/template.html`
- Output: `bundle-size-report/index.html` and `bundle-size-report/data.json` in the current directory

## Rules
- Every number comes from esbuild's metafile. Never invent, estimate, or round-trip a size you did not measure. If the engine cannot bundle the project, say so plainly — do not fabricate a report.
- Read-only against the project. The skill only writes inside `bundle-size-report/`.
- Do not add comments to any command you run.
- The engine needs the project's `node_modules` present. If they are missing, the build cannot resolve imports.

## Step 1 — Find the project

Decide the target from `$ARGUMENTS`:
- A path to a directory → that is the project root.
- A path to a file ending in `.jsx/.tsx/.js/.ts` → the project root is its package directory and that file is the entry override.
- Empty → use the current working directory as the project root.

Confirm it is a frontend project: a `package.json` exists and there is a `src/` entry or an `index.html` pointing at a module script. If you cannot find one, ask the user for the entry file with `AskUserQuestion` rather than guessing.

## Step 2 — Make sure dependencies are installed

The attributor bundles the real code, so the project's imports must resolve. Check for `node_modules`:

```bash
test -d <project>/node_modules && echo present || echo missing
```

If missing, tell the user the engine needs an install first and offer to run it:

```bash
npm --prefix <project> install
```

## Step 3 — Run the attribution engine

Invoke the engine by its installed absolute path so it finds esbuild and the template. It always writes `bundle-size-report/` into the current working directory, so run it from where you want the report. Pass the project root, and optionally an entry file:

```bash
cd <project> && node "$HOME/.claude/skills/bundle-size/scripts/bundlesize.mjs" .
```

With an explicit entry:

```bash
cd <project> && node "$HOME/.claude/skills/bundle-size/scripts/bundlesize.mjs" . src/main.tsx
```

The engine prints a short summary to stdout (heaviest import, app vs vendor split, top ten imports) and writes the full report. If esbuild fails, it explains why — most often a project that needs SCSS/LESS or path-alias resolution the bare bundler does not have. In that case, relay the error; do not produce a partial report.

## Step 4 — Report back

Relay the stdout summary in your own words: name the heaviest import and the KB it added, the minified and gzipped totals, and the app-versus-vendor split. Point the user at the generated site:

```
bundle-size-report/index.html
```

Offer to open it. The page is self-contained — a treemap of imports, a ranked bar list, an app/vendor donut, a sortable table of every import, and a module-level explorer — no server or network needed.

## Notes
- "Minified bytes" is each module's contribution to the minified bundle (`bytesInOutput`), which is the honest unit for attribution. The gzipped total is the whole bundle through zlib; per-import gzip is not reported because gzip is not additive across modules.
- npm packages are grouped by their package name (scoped packages keep their `@scope/`). Everything outside `node_modules` is grouped as `(application code)`.
- The engine measures a production-style build: `bundle`, `minify`, `format=esm`, `NODE_ENV=production`. It is not your project's exact Vite/webpack config, but it is a faithful, repeatable attribution of the same source graph.

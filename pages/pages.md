# GitHub Pages Registry

## Critical Rule

`diegopacheco/ai-playground` has one GitHub Pages deployment workflow:

`.github/workflows/agents-pages.yml`

This workflow must always build and publish the complete `_site` artifact. Every Pages deployment replaces the previous artifact. A second workflow or a partial artifact will remove routes that are not included in the newest deployment.

Never create another workflow that calls `actions/deploy-pages`. Extend the existing aggregate workflow instead.

The `diegopacheco/diegopacheco.github.io` repository is a separate site. Changes to `ai-playground` must never modify or deploy that repository.

That repository has no workflow. It serves `index.html` with classic GitHub Pages from its `master` branch, so pushing `master` publishes it. Its deployment is independent of the `ai-playground` artifact and cannot add or remove any `ai-playground` route.

The `diegopacheco/agent-skills` repository is also a separate site. It has no workflow. It serves `docs/index.html` with classic GitHub Pages from the `/docs` folder of its `main` branch, so pushing `main` publishes it. It cannot add or remove any `ai-playground` route. The `ai-playground` hub pages link to it from their navigation as `Skills`.

## Published Pages

| Repository | Path in repository | Public URL |
|---|---|---|
| `diegopacheco/diegopacheco.github.io` | `index.html` | `https://diegopacheco.github.io/` |
| `diegopacheco/agent-skills` | `docs/index.html` | `https://diegopacheco.github.io/agent-skills/` |
| `diegopacheco/ai-playground` | `agents/landing-page.html` | `https://diegopacheco.github.io/ai-playground/` |
| `diegopacheco/ai-playground` | `agents/landing-page.html` | `https://diegopacheco.github.io/ai-playground/agents/` |
| `diegopacheco/ai-playground` | `sec/index.html` and `sec/landing-page.html` | `https://diegopacheco.github.io/ai-playground/sec/` |
| `diegopacheco/ai-playground` | `games/index.html` and `games/README.md` | `https://diegopacheco.github.io/ai-playground/games/` |
| `diegopacheco/ai-playground` | `games/blockrails/index.html` and `games/blockrails/assets/` | `https://diegopacheco.github.io/ai-playground/games/blockrails/` |
| `diegopacheco/ai-playground` | `games/angry-bode/index.html`, `games/angry-bode/src/`, `games/angry-bode/vendor/` and `games/angry-bode/assets/` | `https://diegopacheco.github.io/ai-playground/games/angry-bode/` |
| `diegopacheco/ai-playground` | `games/seagotchi/index.html` and `games/seagotchi/assets/` | `https://diegopacheco.github.io/ai-playground/games/seagotchi/` |
| `diegopacheco/ai-playground` | `games/gull-about-town/index.html` and `games/gull-about-town/assets/` | `https://diegopacheco.github.io/ai-playground/games/gull-about-town/` |
| `diegopacheco/ai-playground` | `games/dustline/index.html` and `games/dustline/assets/` | `https://diegopacheco.github.io/ai-playground/games/dustline/` |
| `diegopacheco/ai-playground` | `games/jetfun/index.html`, `games/jetfun/vendor/` and `games/jetfun/assets/` | `https://diegopacheco.github.io/ai-playground/games/jetfun/` |
| `diegopacheco/ai-playground` | `mcp/index.html` | `https://diegopacheco.github.io/ai-playground/mcp/` |
| `diegopacheco/ai-playground` | `macos-chrome-apps/index.html` | `https://diegopacheco.github.io/ai-playground/macos-chrome-apps/` |
| `diegopacheco/ai-playground` | `pocs/agentsfile/landing/index.html`, `pocs/agentsfile/logo.svg`, `pocs/agentsfile/spec/SPEC.md` and `pocs/agentsfile/printscreens/` | `https://diegopacheco.github.io/ai-playground/pocs/agentsfile/landing/` |
| `diegopacheco/ai-playground` | `pocs/factoryfile/landing/`, `pocs/factoryfile/logo.svg`, `pocs/factoryfile/spec/SPEC.md` and `pocs/factoryfile/printscreens/render-ui-xp.png` | `https://diegopacheco.github.io/ai-playground/pocs/factoryfile/landing/` |

## Artifact Mapping

The aggregate workflow currently creates these entry points:

| Source | Artifact destination |
|---|---|
| `agents/landing-page.html` | `_site/index.html` |
| `agents/landing-page.html` | `_site/agents/index.html` |
| `agents/logos/` | `_site/logos/` and `_site/agents/logos/` |
| `sec/index.html` | `_site/sec/index.html` |
| `sec/landing-page.html` | `_site/sec/landing-page.html` |
| `games/index.html` | `_site/games/index.html` |
| `games/README.md` | `_site/games/README.md` |
| `games/blockrails/` | `_site/games/blockrails/` |
| `games/angry-bode/` | `_site/games/angry-bode/` |
| `games/seagotchi/` | `_site/games/seagotchi/` |
| `games/gull-about-town/` | `_site/games/gull-about-town/` |
| `games/dustline/` | `_site/games/dustline/` |
| `games/jetfun/` | `_site/games/jetfun/` |
| `mcp/index.html` | `_site/mcp/index.html` |
| `macos-chrome-apps/index.html` | `_site/macos-chrome-apps/index.html` |
| `pocs/agentsfile/landing/` | `_site/pocs/agentsfile/landing/` |
| `pocs/agentsfile/logo.svg`, `pocs/agentsfile/spec/SPEC.md` and the three printscreens it uses | `_site/pocs/agentsfile/` |
| `pocs/factoryfile/landing/` | `_site/pocs/factoryfile/landing/` |
| `pocs/factoryfile/logo.svg`, `pocs/factoryfile/spec/SPEC.md` and `printscreens/render-ui-xp.png` | `_site/pocs/factoryfile/` |

The workflow also copies every media file required by these pages. Source files used by the workflow must be tracked by Git. Ignored build output must not be used as a published asset.

## Updating an Existing Page

1. Edit the page and its tracked assets in the existing source directory.
2. Confirm the directory is present in the `paths` filter in `.github/workflows/agents-pages.yml`.
3. If asset paths changed, update the `Prepare site` step in the same workflow.
4. Preserve every existing artifact entry point and copy operation.
5. Rehearse the complete `Prepare site` step in an isolated temporary directory.
6. Confirm all `ai-playground` entry points exist in the generated `_site` directory.
7. Confirm every local image, video, stylesheet, and script reference resolves inside `_site`.
8. Commit and push the requested source changes and the aggregate workflow change together.
9. Wait for the `Publish Agent Atlas` workflow to finish successfully.
10. Verify every public URL in this document returns HTTP 200 and has the expected page title.

## Adding a Page

1. Create `<route>/index.html` and keep its assets in tracked repository paths.
2. Add `<route>/**` to the `paths` filter in `.github/workflows/agents-pages.yml`.
3. Extend the existing `Prepare site` step to create `_site/<route>/` and copy the page and its assets there.
4. Build the full artifact containing every existing page plus the new page.
5. Never add a separate Pages workflow.
6. Never deploy only the new route.

## POC Counters

Every page that shows a count of POCs must match its source of truth. Whenever a POC is added or removed on any page, recount every counter below and fix the ones that drifted, in the same change.

| Counter | Where it lives | Source of truth |
|---|---|---|
| Agentic AI Projects | `diegopacheco.github.io/index.html` spotlight `Collection 01` | `const projects` plus every `projects.push(` in `agents/landing-page.html` |
| AI POCs on GitHub | `diegopacheco.github.io/index.html` spotlight `Collection 02` and `agents/landing-page.html` `600+` stat | directories in `pocs/`, rounded down to the hundred with `+` |
| MCP Servers | `diegopacheco.github.io/index.html` spotlight `Collection 03` and the `POCs` and `Local dirs` stats in `mcp/index.html` | `const items` in `mcp/index.html`, and its `href` values under `/tree/main/mcp/` |
| Security Projects | `diegopacheco.github.io/index.html` spotlight `Collection 04` and the `Projects` stat in `sec/landing-page.html` | `const projects` in `sec/landing-page.html` |
| AI & CV Games | `diegopacheco.github.io/index.html` spotlight `Collection 05` and the `Games` stat in `games/index.html` | `const games` in `games/index.html` |
| Skills, night agents, patterns | `diegopacheco.github.io/index.html` spotlight `Collection 06` (their sum) and the stats in `diegopacheco/agent-skills` `docs/index.html` | skill folders, `night-agents/skills/`, and `patterns/NN-*` in `diegopacheco/agent-skills` |

Spotlight counts use the exact number with `+`. Page stats use the exact number. A counter change in `diegopacheco.github.io` is pushed from that repository, never from this one.

## Required Checks Before Push

- `git diff --check` passes.
- Every counter in `POC Counters` matches its source of truth.
- The workflow YAML parses successfully.
- Every source and asset copied by the workflow is tracked by Git.
- The complete aggregate build succeeds locally.
- `_site/index.html` exists.
- `_site/agents/index.html` exists.
- `_site/sec/index.html` exists.
- `_site/games/index.html` exists.
- `_site/games/blockrails/index.html` exists.
- `_site/games/seagotchi/index.html` exists.
- `_site/games/gull-about-town/index.html` exists.
- `_site/games/dustline/index.html` exists.
- `_site/games/jetfun/index.html` exists.
- `_site/mcp/index.html` exists.
- `_site/macos-chrome-apps/index.html` exists.
- `_site/pocs/agentsfile/landing/index.html` exists.
- `_site/pocs/factoryfile/landing/index.html` exists.
- No existing route or asset is missing from the artifact.

## Required Checks After Push

- The `Publish Agent Atlas` workflow completed successfully.
- `https://diegopacheco.github.io/` returns HTTP 200.
- `https://diegopacheco.github.io/agent-skills/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/agents/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/sec/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/blockrails/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/angry-bode/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/seagotchi/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/gull-about-town/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/dustline/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/jetfun/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/mcp/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/macos-chrome-apps/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/pocs/agentsfile/landing/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/pocs/factoryfile/landing/` returns HTTP 200.

## Recovery

If a deployment removes existing routes, stop publishing new partial artifacts. Rerun the latest successful `Publish Agent Atlas` workflow to restore the complete artifact, correct the aggregate workflow, and verify every URL before making any further Pages change.

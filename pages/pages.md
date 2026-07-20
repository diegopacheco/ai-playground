# GitHub Pages Registry

## Critical Rule

`diegopacheco/ai-playground` has one GitHub Pages deployment workflow:

`.github/workflows/agents-pages.yml`

This workflow must always build and publish the complete `_site` artifact. Every Pages deployment replaces the previous artifact. A second workflow or a partial artifact will remove routes that are not included in the newest deployment.

Never create another workflow that calls `actions/deploy-pages`. Extend the existing aggregate workflow instead.

The `diegopacheco/diegopacheco.github.io` repository is a separate site. Changes to `ai-playground` must never modify or deploy that repository.

## Published Pages

| Repository | Path in repository | Public URL |
|---|---|---|
| `diegopacheco/diegopacheco.github.io` | `index.html` | `https://diegopacheco.github.io/` |
| `diegopacheco/ai-playground` | `agents/landing-page.html` | `https://diegopacheco.github.io/ai-playground/` |
| `diegopacheco/ai-playground` | `agents/landing-page.html` | `https://diegopacheco.github.io/ai-playground/agents/` |
| `diegopacheco/ai-playground` | `sec/index.html` and `sec/landing-page.html` | `https://diegopacheco.github.io/ai-playground/sec/` |
| `diegopacheco/ai-playground` | `games/index.html` and `games/README.md` | `https://diegopacheco.github.io/ai-playground/games/` |
| `diegopacheco/ai-playground` | `mcp/index.html` | `https://diegopacheco.github.io/ai-playground/mcp/` |
| `diegopacheco/ai-playground` | `macos-chrome-apps/index.html` | `https://diegopacheco.github.io/ai-playground/macos-chrome-apps/` |

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
| `mcp/index.html` | `_site/mcp/index.html` |
| `macos-chrome-apps/index.html` | `_site/macos-chrome-apps/index.html` |

The workflow also copies every media file required by these pages. Source files used by the workflow must be tracked by Git. Ignored build output must not be used as a published asset.

## Updating an Existing Page

1. Edit the page and its tracked assets in the existing source directory.
2. Confirm the directory is present in the `paths` filter in `.github/workflows/agents-pages.yml`.
3. If asset paths changed, update the `Prepare site` step in the same workflow.
4. Preserve every existing artifact entry point and copy operation.
5. Rehearse the complete `Prepare site` step in an isolated temporary directory.
6. Confirm all six `ai-playground` entry points exist in the generated `_site` directory.
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

## Required Checks Before Push

- `git diff --check` passes.
- The workflow YAML parses successfully.
- Every source and asset copied by the workflow is tracked by Git.
- The complete aggregate build succeeds locally.
- `_site/index.html` exists.
- `_site/agents/index.html` exists.
- `_site/sec/index.html` exists.
- `_site/games/index.html` exists.
- `_site/mcp/index.html` exists.
- `_site/macos-chrome-apps/index.html` exists.
- No existing route or asset is missing from the artifact.

## Required Checks After Push

- The `Publish Agent Atlas` workflow completed successfully.
- `https://diegopacheco.github.io/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/agents/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/sec/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/games/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/mcp/` returns HTTP 200.
- `https://diegopacheco.github.io/ai-playground/macos-chrome-apps/` returns HTTP 200.

## Recovery

If a deployment removes existing routes, stop publishing new partial artifacts. Rerun the latest successful `Publish Agent Atlas` workflow to restore the complete artifact, correct the aggregate workflow, and verify every URL before making any further Pages change.

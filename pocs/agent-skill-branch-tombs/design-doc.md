# Branch Graveyard &mdash; design doc

## Goal

Surface the branches a repository has stopped touching, attributed to **who last touched each and when**, and tell the user which are safe to delete. Render it as a self-contained light-theme website with a graveyard metaphor: stale branches are headstones.

## Non-goals

- Deleting or modifying branches. The skill is strictly read-only; it emits delete commands for the user to run, never runs them.
- Guessing intent. It does not try to infer whether an abandoned branch is "important." It reports facts (age, author, merged state) and lets the reader decide.
- A server or live database. The output is one static HTML file plus its JSON.

## Inputs

- A git repository (the current one, or a GitHub repo cloned in a temp folder).
- An optional staleness threshold in days (default 30).

## Definitions

- **Trunk** &mdash; resolved as `origin/HEAD`, else local `main`, else `master`, else the current `HEAD`.
- **Branch** &mdash; any ref under `refs/heads` or `refs/remotes`, excluding the remote `HEAD` pointers. A local branch and its remote twin (same short name) are collapsed into one logical branch; the more recent tip provides the date and author, and the locations are unioned (`local`, `remote`, or `local+remote`).
- **Age** &mdash; whole days between now and the branch tip's committer date.
- **Grave** &mdash; a non-trunk branch whose age exceeds the threshold.
- **Safe grave** &mdash; a grave fully merged into the trunk (`git merge-base --is-ancestor branch trunk` succeeds). Deleting it loses nothing.
- **Unmerged grave** &mdash; a grave with commits ahead of the trunk. Review before deleting.

## Data sources (all deterministic)

| Fact | git command |
|---|---|
| name, short hash, last-touch unix time, author, subject | `git for-each-ref --format=...` over `refs/heads refs/remotes` |
| ahead / behind vs trunk | `git rev-list --left-right --count trunk...branch` |
| merged into trunk | `git merge-base --is-ancestor branch trunk` |
| trunk | `git symbolic-ref --short refs/remotes/origin/HEAD`, with fallbacks |

No value in the report is produced by the model. Two runs on the same repo produce identical output.

## Age tiers

`active` &le; 30, `stale` &le; 90, `abandoned` &le; 365, `ancient` &gt; 365 (days). Tiers drive headstone color; the threshold (independent, default 30) drives whether a branch is a grave at all.

## Output

- `branch-graveyard-report/data.json` &mdash; the full computed model.
- `branch-graveyard-report/index.html` &mdash; `assets/template.html` with the JSON injected at `__GRAVEYARD_DATA__`.

The page renders four regions client-side from the JSON:

1. **Summary cards** &mdash; branches, stale count, safe-to-bury count, unmerged-grave count, oldest grave, and the "gravekeeper" (author with the most stale branches).
2. **The graveyard** &mdash; one headstone per grave, oldest first, with last-touch date, author, age, location, and a merged/unmerged pill. Top edge colored by tier.
3. **Safe to bury** &mdash; per-branch copy-to-clipboard delete commands (`git branch -d` for local, `git push <remote> --delete` for remote) plus a copy-all button.
4. **All branches** &mdash; a sortable, filterable table of every branch including the trunk.

## Engine shape (`scripts/branch_tombs.py`)

`detect_default` &rarr; `list_refs` &rarr; `collapse` (merge local/remote twins) &rarr; per-branch `ahead_behind` + `is_merged` + tier &rarr; `analyze` rolls up the summary &rarr; `render` injects into the template &rarr; `print_summary` to stdout. Standard library only; no third-party dependencies.

## Distribution

- `install.sh` copies `SKILL.md`, the engine, and the template into `~/.claude/skills/branch-tombs`.
- `uninstall.sh` removes that directory.
- `site/index.html` is a light-theme landing page with copy-to-clipboard install/uninstall commands, a feature summary, and usage &mdash; the "install/uninstall website."

## Testing

`sample/build-sample.sh` builds a fixture git repo with eight branches backdated across two years and four authors, mixing merged and unmerged. Running the engine inside it exercises every code path (trunk detection, collapse, ahead/behind, merged detection, all four tiers, safe vs unmerged graves) and produces the report checked into `branch-graveyard-report/`.

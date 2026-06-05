# GitHub Control Panel — Design Doc

## 1. Goal

A central control panel to monitor a set of **public** GitHub repos from one place.
The user supplies a list of repos; the system pulls pull requests, issues, and
contribution data, caches it, and surfaces it through a six-tab dashboard.

## 2. Scope

In scope:
- Aggregate PRs, issues, and contributions across many repos.
- Six-tab UI: Repos, Dashboard, Issues, Action Center, Insights, Settings.
- Cached data served from our own store, refreshed by a user-triggered **Sync**.

Out of scope (for now):
- Write actions back to GitHub (merge, comment, close).
- Private repos / org-wide installs.
- Real-time webhooks (the user syncs on demand instead).
- Multi-user auth on the panel itself.

## 3. Tech Stack

### Frontend
- Node.js + Vite
- React 19
- TypeScript 6.x
- TanStack: Query (data), Router (tabs/routing), Table (grids), Virtual (long lists)
- **Recharts** for Insights charts (decided — see §12)
- Modular, component-first structure
- **Light theme** UI with a blinking green live indicator in the header

### Backend
- Java 25
- Spring Boot 4.0.6
- MySQL with HikariCP
- Spring Data JDBC (no Hibernate / no JPA)
- GitHub access via the JDK `HttpClient` (least libraries) calling the GitHub GraphQL API

### Infra / Local Dev
- MySQL, backend, and frontend all containerized and orchestrated via `podman-compose`
- `start.sh`, `stop.sh`, `test.sh`
- `Containerfile` per service (latest base images, compact)

## 4. High-Level Architecture

```
[React 19 + Vite UI]  --HTTP/JSON-->  [Spring Boot 4 API]  --GraphQL-->  [GitHub]
                                            |
                                            v
                                      [MySQL (cache)]
```

The UI never talks to GitHub directly. A user-triggered Sync makes the backend
pull from GitHub into MySQL; the UI reads only from the backend's cached data,
so it is fast and never hits GitHub rate limits on page load.

(README renders this as an Excalidraw-style hand-drawn diagram, not ASCII.)

## 5. GitHub Integration

- **API**: REST (`/repos/{o}/{r}/pulls`, `/issues`, `/contributors`) — three calls
  per repo, aggregated backend-side. REST is required because GitHub's GraphQL API
  rejects unauthenticated requests; REST is what makes the token optional (see §12.8).
- **Auth — token is never stored.** The panel works unauthenticated (60 req/hr,
  fine for small repo lists). The user may optionally paste a Personal Access
  Token in the **Settings** tab to raise the ceiling to 5,000 points/hr. That
  token is:
  - held only in the UI's in-memory state (no localStorage, no cookies, no disk);
  - sent to the backend only on a Sync request, via an `X-GitHub-Token` header;
  - used transiently for that sync's GraphQL calls and then discarded;
  - **never** written to MySQL, never written to disk, never logged.
  - Reloading the page clears it; the user re-enters it when they want it.
  - If no token is present, sync proceeds unauthenticated.
- **Sync strategy**: on-demand only. The user presses a **Sync** button; there
  is no scheduled/background polling. `POST /api/sync` performs the refresh.
- **What we pull per repo**: PR state (open/closed/merged/draft), author,
  reviewers/requested reviews, CI check status, labels, timestamps,
  mergeability; issue state, assignees, labels, comments, timestamps;
  contribution signals per author.

## 6. Data Model (MySQL, Spring Data JDBC)

- **repository** — id, owner, name, full_name, added_at, last_synced_at
- **pull_request** — id, repository_id, number, title, author, state, draft,
  ci_status, mergeable, created_at, updated_at, closed_at, merged_at, url
- **pr_reviewer** — pull_request_id, login, review_state
- **issue** — id, repository_id, number, title, author, state, body,
  comments_count, created_at, updated_at, closed_at, url
- **label** — id, name, color (+ join tables for pr/issue labels)
- **issue_assignee** — issue_id, login
- **contribution** — repository_id, login, commits, prs_opened, issues_opened,
  reviews

There is **no token table** — the token is never persisted. Aggregations
(counts, medians, trends) are computed from these tables via SQL or in the
service layer.

## 7. Backend Design

### Modules
- `github` — GraphQL client + query builders + response mapping
- `sync` — on-demand refresh orchestration, transient token handling
- `repo` — repo-list CRUD
- `dashboard` / `issues` / `actioncenter` / `insights` — read/query services
- `web` — REST controllers (JSON for the UI)

### REST Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET    | /api/repos              | list tracked repos |
| POST   | /api/repos              | add repos (accepts a list) |
| DELETE | /api/repos/{id}         | remove a repo |
| POST   | /api/sync               | sync now (optional `X-GitHub-Token` header) |
| GET    | /api/dashboard          | aggregated counts/contributions (tab 2) |
| GET    | /api/issues             | all issues across repos (tab 3 left list) |
| GET    | /api/issues/{id}        | single issue detail (tab 3 middle) |
| GET    | /api/action-center      | "needs attention" items (tab 4) |
| GET    | /api/insights           | time series + aggregates (tab 5) |

## 8. Frontend Design

Six tabs via TanStack Router. Shared, modular components (cards, tables,
filters, lists, detail panes). TanStack Query handles fetching/caching/refetch.
A **Sync** button (in the header, available from every tab) triggers
`POST /api/sync` and refetches.

- **Tab 1 — Repos**: input/manage the list of public repos (add as
  `owner/name` or URL, remove, see last-synced time, Sync).
- **Tab 2 — Dashboard**: aggregate view across all repos — issues totals, PRs
  open/closed/pending, contributions. Card + summary-table layout.
- **Tab 3 — Issues**: master-detail. Left = virtualized list of all issues
  across repos (filter by repo/label/state); middle = selected issue rendered
  for reading (title, body, labels, meta).
- **Tab 4 — Action Center** ("Needs My Attention"): cross-repo, cross-resource
  triage inbox — PRs awaiting review, PRs with failing CI, stale issues (no
  activity in 7 days), unassigned issues, PRs open longer than 7 days. Grouped,
  sortable, "what's rotting."
- **Tab 5 — Insights / Analytics**: Recharts over cached data — PR throughput,
  median time-to-merge, issue open-vs-close rate, contributor activity over
  time, stale-PR trend.
- **Tab 6 — Settings**: paste an optional GitHub token (in-memory only, never
  persisted — see §5), plus thresholds/preferences. Clear-token control.

## 9. Caching & Sync

- Source of truth for the UI is MySQL, populated by on-demand sync.
- No scheduled polling — the user controls freshness with the **Sync** button.
- `last_synced_at` shown in the UI so the user knows freshness.

## 10. Local Dev / Ops

- `podman-compose` brings up MySQL, backend, and frontend (all containerized).
- `start.sh` — starts the full stack via `podman-compose`.
- `stop.sh` — stops everything.
- `test.sh` — proves the feature works end to end.
- Config via env vars: DB connection, thresholds. **No token env var** — token
  is entered in the UI at runtime and never stored.

## 11. README Plan (make it cool)

- Punchy overview + the six tabs explained.
- **Excalidraw-style hand-drawn architecture diagram** (wobble filter, "Caveat"
  handwriting font, pastel boxes, solid capture path) — not ASCII.
- **Playwright-captured screenshots** of every tab saved under `printscreens/`
  and rendered inline with short captions.
- Quickstart: `start.sh` / `stop.sh` / `test.sh`, requirements, how to add repos
  and (optionally) paste a token.
- Feature highlights, tech-stack badges, and a short "how sync works" note.

## 12. Decisions (resolved)

1. **Chart library → Recharts.** Best balance for this stack: declarative and
   composable (fits React 19 + component-first), TypeScript types included,
   minimal footprint, and React 19 compatible. visx/ECharts are more powerful
   but heavier and more code; Recharts wins on simplicity.
2. **Token → never stored.** In-memory in the UI, sent per-sync via header, used
   transiently, never on disk/DB/logs. Entered in Settings (Tab 6).
3. **Sync → manual button**, no background polling.
4. **Action Center thresholds → 7 days** (stale issues and PRs-open-too-long).
5. **Contributions → all signals** (commits, PRs opened, issues opened, reviews),
   all-time.
6. **Containerization → all three services** (MySQL + backend + frontend) via
   `podman-compose`.
7. **Theme → light.** The header carries a blinking green dot as a live indicator.
8. **GitHub API → REST, not GraphQL.** GitHub's GraphQL API rejects unauthenticated
   requests (401); only REST honours the 60-requests/hour unauthenticated limit. To
   keep the token genuinely optional (decision #2), the sync layer uses the REST API
   (`/pulls`, `/issues`, `/contributors`). Tradeoff: CI check status is not in the PR
   list payload, so the Action Center "failing CI" group needs a token-gated per-PR
   call to populate (left out of this build); every other signal works without a token.

## 13. Milestones

1. Backend skeleton + MySQL + repo CRUD.
2. GitHub GraphQL client + on-demand sync + data model + transient token.
3. Read endpoints (dashboard, issues, action-center, insights).
4. Frontend shell + TanStack Router tabs + Sync button + Tab 1.
5. Tabs 2–6.
6. Containerfiles + podman-compose + start/stop/test scripts.
7. Cool README with Excalidraw diagram and Playwright screenshots.

# Changelog

Records every user visible change of the agent skills infra: the capabilities, the CLI, the console, the skills, the MCP servers and the scripts.

## 0.2.0 - 2026-11-03

### Added
* The login page lists every local OAuth client on the right with its kind, scopes and secret (show, copy), and signs in as that client in one click.
* An Overview tab with a live, animated topology of every skill and MCP server through the CLI and the capability APIs into Postgres and Loki, where each real state write, log line and eval run travels as a particle, plus a usage matrix and an activity feed.
* Logs, State, Evals and Auth tabs. An app sees only its own data; `infra-admin` sees every app and picks one or all apps with the App filter in the header.
* An `infra-admin` client with read-only admin scopes, and admin APIs to list clients, state entries, log lines and per tenant summaries, plus `GET /v1/me` on the auth service.
* A provisioned Grafana home dashboard with log volume per tenant and level, state keys per tenant, latest eval score per suite, score over time and the audit trail.
* `scripts/traffic.sh` generates traffic non-stop until Ctrl+C, or for a given number of actions, through real MCP stdio sessions to both MCP servers and through both skill scripts, without Claude, and deletes what it created on exit.
* The topology replays each app's real activity from the last hour as faint particles, so it keeps moving between bursts; live changes stay bright.

### Changed
* `agentinfra bootstrap` now only creates new clients or clients whose scopes changed, so existing secrets keep working.
* `start-all.sh` prints Loki's `/ready` link, because Loki has no web page and its root answers 404.
* The evals UI is now the Agent Infra Console; eval runs moved to the Evals tab, and its tenant filter is the shared App filter.

### Fixed
* The topology stopped moving as soon as a traffic run ended, because it only animated changes between polls. It now replays the last hour of real activity, so an app that was active recently keeps showing flow.

### Verified
* `test-all.sh` passed 101 of 101 tests with 0 skipped: 58 infra, 12 MCP, 5 skills, 26 UI, and all 4 eval suites scored 1.0.
* With no traffic running, the Overview kept 1 to 4 replay particles in flight over 10 seconds.
* `traffic.sh` in non-stop mode ran 9 actions with 0 FAIL, stopped on SIGINT with exit 0 and left no bookmarks or decisions behind.
* An app token got 403 on every admin endpoint, `infra-admin` got 403 on every write, and with read-all on, Postgres still refused to update or delete another tenant's rows.
* The State tab filtered to `feature-flags-mcp` showed only its keys, and the Logs and Evals tabs signed in as `decision-log` showed only `decision-log` data.
* During `traffic.sh`, the Overview showed 4 to 16 particles in flight and lit edges.

## 0.1.0 - 2026-11-03

### Added
* OAuth 2 auth service with the client credentials grant, token introspection and server metadata, issuing 15 minute Bearer tokens whose subject is the tenant.
* `agentinfra bootstrap` creates one OAuth client per skill, MCP server, the evals UI and the tests, with hashed secrets and git ignored credentials files.
* State service: a private JSON key/value store per tenant, isolated by the service and again by Postgres Row Level Security.
* Logs service: structured logs per tenant in Grafana Loki, with Loki tenants for isolation and a provisioned Grafana for browsing all tenants.
* Evals service with 9 common evaluators (exact match, contains, not contains, regex, JSON valid, JSON field equals, min and max length, max latency) and stored runs per tenant.
* Evals UI with OAuth sign in, totals, current health per tenant suite, a sortable runs table filtered by tenant, run details with every check, and the evaluator catalog.
* `agentinfra` CLI with `token`, `state`, `log` and `eval` commands, a token cache and one retry on an expired token.
* `decision-log` skill to record, list, show, supersede and remove architecture decisions across sessions.
* `answer-grader` skill to grade an answer file against checks, keep the score history and best score, and report the trend.
* `bookmarks-mcp` MCP server with save, list, search and delete bookmark tools.
* `feature-flags-mcp` MCP server with set, get, list, evaluate and delete flag tools, stable rollout buckets and an audit log of every change.
* An eval suite for each skill and MCP server, run by `scripts/test-all.sh`.
* Scripts `setup.sh`, `start-all.sh`, `stop-all.sh`, `status.sh`, `test-all.sh`, `ui.sh`, `sql-console.sh` and `install.sh`, which adds the skills and MCP servers to this project only.

### Fixed
* `json_field_equals` passed when a field held `1` and the check expected `true`, because Python treats `1` and `True` as equal. Booleans are now compared by type, so a wrong answer fails the check.
* Grafana showed no log lines because Loki rejected queries for several tenants at once. Loki now allows multi tenant queries for the Grafana admin view, while the logs API still sends exactly one tenant.
* Stopping the stack deleted all state, eval runs and logs, because the containers had no volumes. Postgres and Loki data now live in named volumes and survive `stop-all.sh`.

* `start-all.sh` took about 15 seconds at "starting postgres, loki and grafana", because Loki waits 15 seconds after start and then waits again for its query workers to find a scheduler. Loki now skips that delay and its workers connect directly, so `start-all.sh` finishes in about 3 seconds.

### Verified
* `start-all.sh` after `stop-all.sh` took 2.7 seconds, and `test-all.sh` still passed 73 of 73 tests with every eval suite at 1.0.
* `scripts/test-all.sh` passed: ty check clean, 49 infra tests, 12 MCP tests, 5 skill tests and 7 UI tests, 73 in total with 0 failures and 0 skipped.
* The 4 eval suites scored 1.0: decision-log 6 of 6, answer-grader 4 of 4, bookmarks-mcp 5 of 5, feature-flags-mcp 5 of 5.
* A tenant reading another tenant's key got 404, its log query returned none of the other tenant's lines, and a read only client got 403 on a write.
* As the app role in Postgres, no tenant setting returned 0 rows and inserting a row for another tenant was refused.
* After `install.sh`, Claude Code listed the `decision-log` and `answer-grader` skills.

# Admin Console

A single web console for **Cassandra, MySQL, Postgres, Redis, etcd, Kafka and Elasticsearch** — as many servers as you want, grouped into projects.

Every console works the same way: browse the schema on the left, write a query in the middle, read results at the bottom. Every statement is **read-only** and **audited**.

![Architecture](printscreens/architecture.png)

## What it does

- **Seven engines, one interface.** The same three-pane console drives all of them. Adding an eighth is one backend class plus one frontend descriptor.
- **Read-only, enforced three ways.** A statement guard, a read-only connection inside a rolled-back transaction, and a `SELECT`-only database user. All three are tested independently.
- **Full audit trail.** Who ran what, when, against which server, how long it took, and — most usefully — every statement that was *rejected* and why.
- **Encrypted connection secrets.** AES-256-GCM, so passwords never appear in a `SELECT`, a screenshot or a log line.
- **Real users and roles.** `admin` manages config, users and the audit trail; `user` can query.
- **AI query authoring.** Describe what you want, and a local agent CLI (`claude -p`, `codex exec`, `agy -p`) writes the query in the right grammar for that engine. Suggestions are loaded into the editor for review — nothing runs on its own.
- **Keyboard first.** `⌘K` jumps to any page, a searchable picker switches connections, `⌘↵` runs the query, double-click any row for the full record.

## Quick start

```bash
./start.sh            # metadata postgres + backend + frontend, then prints all links
./demo/demo-start.sh  # 7 seeded target servers, auto-registered as the "demo" project
./links.sh            # every URL
./stop.sh             # stop everything — your data is kept
./destroy-all.sh      # purge everything, including all data (asks for confirmation)
```

`start.sh` is idempotent and `stop.sh` is non-destructive: your projects, connections, users,
saved queries and audit trail live in a named volume and survive restarts. `destroy-all.sh` is
the only script that deletes data, and it makes you type `DESTROY` to confirm.

Then open <http://localhost:4321> and log in with **admin / admin**.

| URL | What |
|---|---|
| `http://localhost:4321/` | consoles |
| `http://localhost:4321/projects` | projects and connections |
| `http://localhost:4321/audit-trail` | audit trail |
| `http://localhost:4321/users` | user management |
| `http://localhost:4321/settings/ai` | AI query authoring |
| `http://localhost:8099/swagger` | Swagger UI |
| `http://localhost:6006/` | Storybook (`cd frontend && bun run storybook`) |

## The console

Browse tables on the left, write a query, page through results. `⌘↵` runs it.

![Console](printscreens/01-console-postgres.png)

### Pick a connection

A searchable picker rather than tabs — tabs stop scaling once you have more than a handful of servers. Search matches connection name, engine, **and host**, because `prod` vs `staging` is usually a host distinction.

![Engine picker](printscreens/02-engine-picker.png)

### Go anywhere with ⌘K

Everything fits without scrolling. `⌘K → Consoles` chains straight into the connection picker.

![Command palette](printscreens/03-command-palette.png)

### Inspect a row

**Double**-click a row — not single, so selecting and copying cell text still works. JSON values are pretty-printed; `↑↓` walks rows without closing.

![Row detail](printscreens/04-row-detail.png)

### Save queries for the project

Shared with everyone on the project, optionally pinned to one connection — or left loose so the same SQL runs against staging and prod.

![Saved queries](printscreens/05-saved-queries.png)

## Every engine, same console

Kafka: topics and partitions on the left, a bounded `consume` in the middle. The console **never joins a consumer group and never commits an offset**, so it cannot disturb your real consumers.

![Kafka console](printscreens/06-console-kafka.png)

Redis: keys badged by type; folding a hash shows its fields.

![Redis console](printscreens/07-console-redis.png)

## Read-only means read-only

Writes are refused with an engine-appropriate reason, and nothing renders that could be mistaken for success.

![Read-only denial](printscreens/08-read-only-denied.png)

| Engine | Allowed | Rejected |
|---|---|---|
| MySQL / Postgres | `SELECT`, `SHOW`, `DESCRIBE`, `EXPLAIN` | writes, DDL, a second `;` statement, writes hidden in a CTE |
| Cassandra | `SELECT` | all DML, DDL, `TRUNCATE` |
| Redis | commands the server itself flags `readonly` | writes, admin, `EVAL` (Lua can write) |
| etcd | `get`, `get --prefix`, `range` | `put`, `del`, `txn`, `compact` |
| Kafka | `list`, `describe`, `offsets`, bounded `consume` | produce, topic admin, offset resets |
| Elasticsearch | `GET`/`HEAD` on `_search`, `_count`, `_mapping`, `_cat` | every other verb, plus `_bulk`, `_delete_by_query`, `_reindex` **even as `GET`** |

## Audit trail

Grouped by query, so paging through 6 pages reads as one entry. Denials are highlighted with their reason.

![Audit trail](printscreens/09-audit-trail.png)

## Projects and connections

Connections are stored encrypted. The form tells you to use a `SELECT`-only database account, because that's the layer that still holds if the guard is ever bypassed.

![Projects](printscreens/10-projects.png)

## AI query authoring

Pick which agent CLI writes your queries. The choice is remembered per user, in the database, so it follows you across machines.

![AI settings](printscreens/11-ai-settings.png)

**What is sent:** the engine's grammar, your schema **names** (tables, columns, keys, topics, fields), and your request.
**What is never sent:** credentials, hostnames, or any row of data.

Suggestions pass the same read-only guard as anything you type, and never execute on their own.

## Users

![Users](printscreens/12-users.png)

## Design system

Every component has stories and tests.

![Storybook](printscreens/13-storybook.png)

## Stack

**Backend** — Java 25, Maven, Spring Boot 4.1.0 on virtual threads, HikariCP (one dynamic pool per connection), Spring JDBC, Postgres + MySQL drivers, Lettuce, DataStax driver, jetcd, kafka-clients. Elasticsearch uses the JDK's `HttpClient`, so it needs no extra dependency.

**Frontend** — Astro 7, React 19 islands, TypeScript 7, bun, CodeMirror 6, Storybook 10, Jest 30.

**Storage** — one Postgres holds projects, connections, users, saved queries, the audit log and the encryption key.

## Testing

```bash
cd backend && mvn test        # unit tests only — integration tests are excluded
cd frontend && bun run test   # jest
./it.sh                       # brings up all 7 servers, then runs integration tests
```

Integration tests are tagged `integration-test` and excluded from the normal build, so `mvn test` and `mvn package` never need a database. `it.sh` flips the tag on.

Every integration test runs against a **real** server — no mocked databases. That includes paging correctness on all seven engines (page 2 continues exactly where page 1 ended), writes being rejected by the database itself when the guard is bypassed, and a test that consumes 50 Kafka messages then asserts a real consumer group's committed offsets are **byte-identical** before and after.

## Security

What this commits to:

- Read-only everywhere, in three independent layers, with no UI toggle to disable it
- Connection secrets encrypted with AES-256-GCM
- PBKDF2 password hashing, two roles, no hardcoded credential (`admin/admin` is a first-boot seed the UI nags you to change)
- Append-only audit of every statement, allowed or denied
- Backend binds `127.0.0.1` by default
- Secrets are never returned by any API, never logged, never in audit rows

What it does **not** do — stated plainly rather than left implicit:

- **The encryption key lives in the same Postgres as the ciphertext.** A `pg_dump` contains both halves. This protects against casual disclosure, not against someone who obtains the database. Moving the key to an env var is a one-class change (`MasterKeyProvider`).
- No TLS between browser and backend — loopback-only deployment is the mitigation
- No login rate limiting
- No audit-log tamper evidence (hash chaining)
- No MFA, SSO, or per-connection ACLs
- **AI sends schema names off the machine.** Opt-in per user, disableable globally, flagged in the UI at the point of use.

Do not deploy this to a shared or public host as-is.

## Docs

[`design-doc.md`](design-doc.md) has the full design: why Postgres is the system of record, why cursors are forward-only (Cassandra's paging state forces it), why modals portal to `document.body`, and a decisions log recording every choice and its reasoning.

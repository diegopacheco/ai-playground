# Generic Admin Console — Design Doc

Status: draft for approval. Nothing is built yet.

## 1. What this is

A single web console that connects to **any number** of Cassandra, MySQL, Postgres, Redis, etcd, Kafka and Elasticsearch servers, grouped into **projects**, and gives each one the same three-pane workflow: browse the schema on the left, write and run a **read-only** query in the middle, read the result at the bottom. Every statement anyone runs is recorded in an audit trail.

It is a generalization of the admin section of `simple-system-design/social-network-java25`. That project hardcodes one MySQL, one Cassandra and one Redis into `AdminDataService`, and its schema queries hardcode the `social_network` keyspace name. Here every connection is user-supplied at runtime, stored encrypted, and access-controlled.

## 2. Assumptions

Stated up front so you can correct them before any code exists.

1. Results are **server-side paginated** (your call, resolving the "one page" wording). One result grid sized to the bottom pane, 100 rows per page, with page controls in the pane footer. The grid itself never scrolls the browser window — "all fit in one page" is satisfied by the pane being self-contained, not by capping the result set. See 4.3 for why the controls are Prev/Next rather than jump-to-page.
2. Multi-project means: a project is a named bag of connections (e.g. project `social-network` holds its MySQL + Cassandra + Redis). You switch projects in the header; the left panel then lists that project's connections.
3. `admin/admin` is the **bootstrap** account, seeded on first boot only. After that, users are managed in the UI. It is not a hardcoded backdoor.
4. Read-only is enforced **server-side** and is not user-toggleable. There is no "I know what I'm doing" escape hatch in this iteration.

## 3. Six conflicts between the requested stack and a *generic, secured* console

These are the parts I'd push back on. Each has a recommendation.

### 3.1 "Encrypted configs in the FS" vs "store configs in Postgres" — resolved: Postgres

You asked for both; they're mutually exclusive as a system of record. **Decided (confirmed): Postgres is the system of record — everything lives there.** Projects, connections, users and the audit trail. You need Postgres regardless for the audit trail — an append-only log of every query by every user is a database workload, not a JSON-file workload, and concurrent users would corrupt a file.

The "encrypted at rest" requirement is honored by **envelope encryption of the secret fields** (passwords, TLS keys) inside those Postgres rows — see 3.2. So config is encrypted *and* in Postgres; what's dropped is the plain-JSON-file-on-disk store from the original brief.

The one thing that cannot live in Postgres is the console's *own* Postgres connection — that bootstraps from `application.yml` / env vars.

### 3.2 Key storage — decided: Postgres, alongside the ciphertext

**Decided: SQLite is dropped. Everything, including the master key, lives in Postgres.** One store, one dependency-free crypto path, no extra moving part.

- **Postgres** holds connection rows whose secret fields are AES-256-GCM ciphertext, and a `keys` table holding the master key, generated on first boot.
- Encryption is JDK-only (`javax.crypto`, AES/GCM/NoPadding, random 96-bit IV per value, IV prepended to ciphertext). Zero dependencies. `org.xerial:sqlite-jdbc` is no longer needed.

**What this protects against, stated honestly so nobody later assumes more:** with the key sitting next to the ciphertext, a single `pg_dump` contains both halves. Anyone who can read the database can decrypt every stored password. So this defends against **casual disclosure** — someone glancing at a `SELECT * FROM connections`, a screenshot, a log line, a `\d` in psql — and not against anyone who takes the database.

Concretely, the "encrypted at rest" property in §9 is therefore **obfuscation, not a security boundary**. That's a deliberate, informed trade for simplicity, not an oversight. If you later want it to be a real boundary, the change is small and localized: move the key out of Postgres into an env var read at boot (`MasterKeyProvider` is a one-method interface precisely so this stays a one-class swap), at which point stolen dumps, replicas and backups become useless without it.

The same `keys` table holds the JWT signing secret, for the same reason.

### 3.3 Spring Data JDBC assumes one static datasource — decided: dynamic pools

`spring-boot-starter-data-jdbc` autoconfigures **one** `DataSource` from `application.yml` and wires repositories to it at startup. That is exactly right for the console's own Postgres metadata store — but wrong for target connections, which are created on demand from encrypted DB rows.

**Decided (confirmed):** the autoconfigured `DataSource` + Spring Data JDBC repositories serve the **metadata store only**. Target connections get **dynamic HikariCP pools** — `ConnectionRegistry` builds and caches one `HikariDataSource` per configured JDBC connection, lazily on first query, evicting and closing it when the config changes.

Because these pools are created at runtime from user input rather than from static config, the registry owns their whole lifecycle:

- **Sizing** — small per pool (`maximumPoolSize` 4, `minimumIdle` 0), since a console issues interactive queries, not load. Twenty configured connections then cost at most 80 sockets, not 20 × default-10.
- **Idle eviction** — a pool with no query for 10 minutes is closed entirely, not just idled down. An admin console spends most of its life with nobody looking at it, and holding connections open against twenty production databases overnight is antisocial.
- **Failure isolation** — `connectionTimeout` 5s and `initializationFailTimeout` -1, so an unreachable target surfaces as one broken pane instead of blocking startup or wedging the registry for every other connection.
- **Eviction correctness** — editing or deleting a connection closes its pool and invalidates any open result cursors (4.3) bound to it, so a config change can never serve rows from the old target.

Same reasoning for Redis and Cassandra: use the **raw Lettuce client and raw DataStax `CqlSession`** for targets, not Spring Data Redis / Spring Data Cassandra, whose autoconfiguration binds one server from properties. Both are cached and evicted by the same registry on the same rules.

Net effect on your stack list: everything stays. Spring Data JDBC just points at the console's own database instead of the targets.

### 3.4 etcd has no tables or columns

etcd is a flat, ordered key→value store with no schema. "List all tables, all columns (foldable)" has no meaning there.

**Recommendation:** the left panel shows a **prefix tree** synthesized by splitting keys on `/` — so `/config/app/db` folds under `/config` → `/app` → `db`. Folding a leaf shows value, revision and lease. The middle pane accepts an etcdctl-style grammar restricted to reads (`get`, `get --prefix`, `range`), consistent with how the Redis pane accepts Redis commands.

etcd speaks gRPC/protobuf, so **jetcd** is unavoidable.

Related: Redis keys have types, not columns. The left panel lists keys with a type badge (the reference project's `redisAll()` already returns `{key, type}` pairs). Folding a key renders per type: hash → field/value table, list/set/zset → members, string → value, stream → recent entries.

### 3.5 Kafka — the console must be invisible to the cluster

Kafka fits the tree and the grid better than I first assumed. `cluster → topic → partition` is a natural `SchemaNode`, with `detail` carrying leader, replicas, ISR, earliest/latest offset and lag; consumer groups form a second root. A bounded consume is genuinely a table: columns `[partition, offset, timestamp, key, value, headers]`. And Kafka is **offset-native**, so it's the cleanest fit of all seven engines for cursor paging (4.3) — the cursor is literally a partition→next-offset map.

**The one thing that must not be got wrong:** a console that consumes carelessly will perturb the cluster it is inspecting. If it joins a consumer group, it triggers a rebalance of that group's real consumers. If it commits offsets, it corrupts their progress and can skip unprocessed messages in production. Either would make this tool actively dangerous on a live system, and both are the *default* behavior of a naively configured consumer.

So the Kafka engine is pinned to observer semantics, non-negotiably:

- `assign()` to explicit partitions — **never** `subscribe()`, so no group is ever joined and no rebalance is ever triggered.
- `enable.auto.commit=false`, and no code path calls `commitSync`/`commitAsync`. The consumer has no group to commit to anyway.
- A unique random `client.id` per query, so the console is identifiable in broker logs and never collides with a real client.
- Bounded reads only: every consume has an explicit start (`earliest`/`latest`/offset/timestamp) and a row limit, with `max.poll.records` and a poll timeout — no open-ended tailing that pins a broker connection.

Grammar (reads only): `list topics`, `describe topic <t>`, `list groups`, `describe group <g>`, `offsets <t>`, and `consume <t> [--partition N] [--from earliest|latest|offset N|timestamp T] [--limit N]`.

Rejected: `produce`/`send`, topic create/delete/alter, config alter, `delete records`, consumer-group offset reset, and group deletion. Read-only here protects other people's runtime state, not just data.

Dependency: `kafka-clients` (`AdminClient` for the tree, `KafkaConsumer` for reads). One new dependency.

### 3.6 Elasticsearch — no new dependency needed

`index → mapping field` maps onto `SchemaNode` directly, with the field type in `detail` and object/nested fields folding naturally. Aliases and `_cat/indices` metadata form the tree's second root.

The middle pane works like Kibana's Dev Tools, which is the interface people already know: `GET /index/_search` followed by a JSON body, plus shorthands `GET /_cat/indices`, `GET /index/_mapping`, `GET /index/_count`.

**Read-only is enforced by method and endpoint, not by parsing JSON:** only `GET`/`HEAD` verbs are accepted, and `_bulk`, `_update_by_query`, `_delete_by_query`, `_reindex`, `_close`, `_open` and any index-management endpoint are rejected outright even when someone dresses them as a `GET` — several of those genuinely accept `GET` and would happily mutate. Whitelisting endpoints is the reliable check; inspecting the body is not.

Paging uses **`search_after` with a point-in-time**, not `from`/`size`. `from` breaks past `index.max_result_window` (10k by default) and gets more expensive the deeper you go — precisely the "browse a big index" case a console is for. The cursor carries the PIT id plus the last sort values, and the PIT is released when the cursor expires or the user closes the tab. Aggregation results come back as a second table rather than being flattened into hits.

Result mapping: columns are `_id`, `_score`, `_index` plus the union of `_source` keys across the page, with nested objects rendered as dotted paths.

**No new dependency.** Elasticsearch is plain REST over HTTP, so the JDK's `HttpClient` plus the Jackson that Spring Boot already ships is enough. The official `co.elastic.clients` library would pull a large transitive tree to do what four HTTP calls do, so it's deliberately skipped — this is one of the rare cases where your least-libraries rule and the pragmatic choice agree completely.

## 4. Architecture

```mermaid
flowchart TB
  subgraph FE["Frontend — Astro 7 + React islands (bun, port 4321)"]
    DS["design-system<br/>Storybook"]
    CON["ConsolePane island"]
    AUD["AuditTrail + Users"]
    DS --> CON & AUD
  end

  subgraph BE["Backend — Spring Boot 4.1.0, Java 25, MVC on virtual threads (port 8099)"]
    AUTH["AuthFilter<br/>HMAC JWT cookie"]
    API["ConsoleController"]
    GUARD["ReadOnlyGuard<br/>rejects writes"]
    AUDIT["AuditService"]
    REG["ConnectionRegistry"]
    AUTH --> API --> GUARD --> AUDIT --> REG
  end

  subgraph ENGINES[" "]
    E1["JdbcEngine<br/>MySQL + Postgres"]
    E2["CassandraEngine"]
    E3["RedisEngine"]
    E4["EtcdEngine"]
  end

  META[("Postgres :5433<br/>projects, connections, users,<br/>audit_log, keys")]
  TARGETS[("target servers")]

  CON & AUD -->|"fetch, credentials: include"| AUTH
  REG --> E1 & E2 & E3 & E4
  API <--> META
  REG -->|"decrypt secrets"| META
  E1 & E2 & E3 & E4 -->|"read-only"| TARGETS
```

### 4.1 The Engine SPI — the thing that makes it generic

Every console pane in the UI is the same React component. That only works if every backend engine returns the same shapes:

```java
public interface Engine {
    String kind();
    SchemaTree schema(ConnectionConfig config);
    QueryResult query(ConnectionConfig config, String statement, PageRequest page);
    Health ping(ConnectionConfig config);
    void assertReadOnly(String statement);
}

public record SchemaNode(String name, String kind, String detail, List<SchemaNode> children) {}
public record SchemaTree(List<SchemaNode> roots) {}
public record PageRequest(int size, String cursor) {}
public record QueryResult(List<String> columns, List<Map<String,Object>> rows, long elapsedMs,
                          int pageNumber, String nextCursor, boolean hasMore, Long totalRows) {}
```

`SchemaNode` is deliberately recursive, because the engines nest differently: JDBC is `schema → table → column`, Cassandra is `keyspace → table → column` (with partition/clustering key markers in `detail`), Redis is `key → field`, etcd is `prefix → prefix → key`. One tree component renders all four.

`QueryResult` is already column/row shaped for SQL and CQL. Redis and etcd responses are normalized into it — `KEYS` becomes columns `[key, type]`, a hash becomes `[field, value]`, an etcd range becomes `[key, value, revision]`. This is the main new work versus the reference project, which returns a bare `Object` for Redis and makes the frontend guess.

Adding a sixth engine later = one class + one autocomplete token file, plus a paging strategy (4.3).

### 4.2 Read-only enforcement — three independent layers

A parser alone is not enough; anything that only inspects a string can eventually be worked around. Defense in depth:

**Layer 1 — statement guard (`assertReadOnly`), per engine:**

| Engine | Allowed | Rejected |
|---|---|---|
| MySQL / Postgres | `SELECT`, `SHOW`, `DESCRIBE`, `EXPLAIN` (without `ANALYZE`), `WITH` whose body is a `SELECT` | everything else, **any** `;`-separated second statement, and CTEs containing `INSERT`/`UPDATE`/`DELETE` |
| Cassandra | `SELECT` only | all DML, DDL, `TRUNCATE`, `USE` |
| Redis | commands whose `COMMAND INFO` flags include `readonly` and exclude `write`/`admin` | everything else, plus `EVAL`/`EVALSHA`/`FCALL` (Lua can write), `SUBSCRIBE`, `MONITOR` |
| etcd | `get`, `get --prefix`, `range` | `put`, `del`, `txn`, `compact`, lease and auth operations |

Redis uses the server's own command metadata rather than a hand-maintained list, so it stays correct as Redis adds commands.

**Layer 2 — connection-level read-only:** JDBC connections open with `setReadOnly(true)`, and Postgres sessions additionally set `default_transaction_read_only=on`, so the *server* rejects writes even if the parser is fooled. Queries run inside a rolled-back transaction with a statement timeout (30s default).

**Layer 3 — credentials:** the docs and the connection form both state that connections should use a database user granted only `SELECT`. The console cannot enforce this, but it's the layer that actually holds if 1 and 2 both fail, so it's surfaced in the UI rather than buried in a README. `demo/` sets up exactly such users, so the default path models the right thing.

A rejected statement is still written to the audit trail with `allowed=false` and the reason. Denied attempts are the most interesting rows in the log.

### 4.3 Pagination — cursor-based, because Cassandra forces it

Server-side paging, 100 rows per page. The awkward part is that no two engines page the same way, and one of them can't do random access at all.

| Engine | Mechanism | Random access? |
|---|---|---|
| MySQL / Postgres | scrollable `ResultSet` held open in a read-only transaction; the cursor token identifies the held statement + absolute row offset | yes, but see below |
| Cassandra | native driver paging state (`ExecutionInfo.getSafePagingState()`), passed back as the cursor | **no — forward-only** |
| Redis | `SCAN` cursor for key listing; materialized values (hash fields, list members) sliced in memory | yes |
| etcd | `range` with `limit` + start-key, the cursor being the last key seen | forward-only in practice |

Cassandra's paging state is opaque and strictly forward-only — there is no "jump to page 47" without re-scanning from the start, which on a large table is exactly the query you don't want an admin console issuing casually.

**Decision: cursor paging with Prev/Next controls, not numbered page jumps.** The pane footer shows `page N · 100 rows · 42ms` with Prev/Next/First. This is the honest common denominator; offering a page-number box that silently degrades to a full re-scan on four of seven engines would be worse than not offering it.

`totalRows` is nullable and populated **only when it's cheap**: never for Cassandra (counting means scanning), never for etcd, and for SQL only when the user opts in via a "count rows" button that runs a separate `SELECT count(*)` over the statement as a subquery. Guessing a total by scanning is the classic way an admin console takes down the database it's inspecting.

Held cursors are resources, so `ConnectionRegistry` expires them after 5 minutes idle and caps concurrent open cursors per user. An expired cursor returns `410 Gone` and the UI offers to re-run the statement rather than silently returning wrong rows.

### 4.4 Data model (Postgres, port 5433)

```sql
keys(id, purpose unique, key_material bytea, created_at)
users(id, username unique, password_hash, password_salt, role, enabled, created_at, last_login_at)
projects(id, name unique, created_at, created_by)
connections(id, project_id fk, name, kind, host, port, database, keyspace, datacenter,
            username, secret_ciphertext bytea, options_json, created_at, created_by)
audit_log(id, query_id, page, at, username, connection_id, project_id, kind, statement,
          allowed, denial_reason, elapsed_ms, row_count, error, client_ip)
```

**Paging and the audit trail.** You flagged that paging means more audit rows, so the model separates the two: a statement execution gets a fresh `query_id` and `page=1`; each subsequent page fetch writes a row with the **same** `query_id` and an incrementing `page`. `/audit-trail` groups by `query_id` by default and shows `page 1 of 6` on the group, with an expander for per-page timing — so browsing a result set reads as one entry, not six, while the detail is still there for forensics. Denials never have follow-on pages.

`keys` holds two rows — `master` (secret encryption) and `jwt` (token signing) — both generated on first boot. `secret_ciphertext` is AES-256-GCM over a small JSON blob (`{"password": "...", "tlsKey": "..."}`), so adding future secret fields doesn't need a migration. `audit_log` is append-only — no `UPDATE`/`DELETE` grants for the application role — and indexed on `(at desc)`, `(username, at desc)` and `(connection_id, at desc)`.

Schema is created at startup from `schema.sql`, following the reference project's approach.

Password hashing is **PBKDF2WithHmacSHA256**, 600k iterations, 16-byte random salt, via the JDK — no Spring Security dependency, consistent with the dependency-free HMAC JWT below.

### 4.5 Backend package layout

```
backend/src/main/java/com/github/diegopacheco/adminconsole/
  AdminConsoleApplication.java
  auth/        Jwt, AuthFilter, AuthController, PasswordHasher, CurrentUser
  user/        User, UserRepository, UserService, UserController, BootstrapAdmin
  crypto/      MasterKeyProvider, PostgresKeyStore, SecretCipher (AES-GCM)
  config/      OpenApiConfig, WebConfig, VirtualThreadConfig, MetadataDataSourceConfig
  project/     Project, ConnectionConfig, ProjectRepository, ConnectionRepository, ProjectController
  registry/    ConnectionRegistry, PoolKey, CursorCache, CursorExpiredException
  audit/       AuditEntry, AuditRepository, AuditService, AuditController, HistoryController
  engine/      Engine, EngineRegistry, SchemaTree, QueryResult, Health, ReadOnlyViolation
  engine/jdbc/      JdbcEngine, JdbcSchemaReader, SqlReadOnlyGuard
  engine/cassandra/ CassandraEngine, CqlReadOnlyGuard
  engine/redis/     RedisEngine, RedisCommandParser, RedisReadOnlyGuard, RedisValueReader
  engine/etcd/      EtcdEngine, EtcdCommandParser, EtcdReadOnlyGuard, PrefixTreeBuilder
  console/     ConsoleController, QueryRequest, ApiExceptionHandler
```

**Spring MVC on virtual threads, not WebFlux.** The reference project is WebFlux and wraps every blocking driver call in `Mono.fromCallable(...).subscribeOn(Schedulers.boundedElastic())` — ceremony that buys nothing, since JDBC, `CqlSession.execute` and jetcd all block anyway. On Java 25 with `spring.threads.virtual.enabled=true` the same code is plain synchronous methods. Fewer moving parts, simpler tests, and wrapping every query in audit timing is straightforward `try-finally` rather than reactive operators. Flagging this as a deliberate deviation from the reference.

### 4.6 HTTP API

| Method | Path | Role | Purpose |
|---|---|---|---|
| `POST` | `/api/auth/login` | — | username/password → JWT cookie |
| `POST` | `/api/auth/logout` | user | clear cookie |
| `GET` | `/api/auth/session` | user | current user and role |
| `POST` | `/api/auth/password` | user | change own password |
| `GET` | `/api/users` | admin | list users |
| `POST` | `/api/users` | admin | create user |
| `PUT` | `/api/users/{id}` | admin | role / enabled |
| `POST` | `/api/users/{id}/password` | admin | reset password |
| `DELETE` | `/api/users/{id}` | admin | delete user |
| `GET` | `/api/projects` | user | projects and connections (secrets never returned) |
| `POST` `PUT` `DELETE` | `/api/projects/{id}` | admin | manage projects |
| `POST` `PUT` `DELETE` | `/api/projects/{id}/connections/{cid}` | admin | manage connections |
| `GET` | `/api/connections/{cid}/ping` | user | health check |
| `GET` | `/api/connections/{cid}/schema` | user | `SchemaTree` |
| `GET` | `/api/connections/{cid}/node?path=` | user | lazy-expand one node |
| `POST` | `/api/connections/{cid}/query` | user | `{statement, pageSize}` → first page + `query_id` + cursor |
| `GET` | `/api/connections/{cid}/query/{queryId}?cursor=` | user | next page; `410` if the cursor expired |
| `DELETE` | `/api/connections/{cid}/query/{queryId}` | user | release a held cursor |
| `POST` | `/api/connections/{cid}/query/count` | user | opt-in `SELECT count(*)`, SQL only |
| `GET` | `/api/history?connection=&limit=` | user | **own** recent distinct statements |
| `GET` | `/api/audit?user=&connection=&allowed=&from=&to=&page=` | admin | audit trail, grouped by `query_id` |
| `GET` | `/api/audit/export.csv` | admin | audit export |
| `GET` | `/swagger` | user | redirect → `/swagger-ui/index.html` |

Two roles: `admin` (everything) and `user` (query and read schema; no config, no audit, no user management). One controller family serves all seven engines — the reference project's separate `/mysql/*`, `/cassandra/*`, `/redis/*` route families are what forces per-engine frontend code.

Connection responses **never** include secrets, not even masked ciphertext.

`/api/history` is scoped to the caller — it reads `audit_log` filtered to their own username, server-side, never by a client-supplied user parameter. A `user` seeing another user's queries would be an audit-trail leak through the back door, so the endpoint takes no user argument at all.

### 4.7 Auth

Built on the reference project's `AdminJwt` + `AdminWebFilter`, which are dependency-free HMAC-SHA256 and work well, extended for real users:

- Login checks the `users` table with PBKDF2 and a constant-time comparison, then issues an 8h HS256 token carrying `sub` (username) and `role`, set as an `HttpOnly` `SameSite=Lax` cookie `admin_console_token`.
- A `OncePerRequestFilter` at highest precedence rejects unauthenticated requests to `/api/**` (except `/api/auth/login`), `/swagger*` and `/v3/api-docs*` with `401`, and role-violating requests with `403`.
- The JWT secret is generated on first boot into the `keys` table — no committed default, and it survives restarts so sessions aren't silently invalidated.
- `BootstrapAdmin` seeds `admin`/`admin` **only when the users table is empty**, and the UI shows a persistent banner until that password is changed.

## 5. Frontend

### 5.1 Structure

Astro 7 renders the shell and routes; every interactive surface is a React island hydrated with `client:load`. Astro's dev server *is* Vite, so "vite + astro" is satisfied by Astro's built-in pipeline rather than a second config.

```
frontend/
  src/
    design-system/          Storybook lives here — zero app knowledge
      tokens/               colors.ts, spacing.ts, typography.ts
      Button/ Badge/ Panel/ Split/ Tree/ DataGrid/ Field/ Select/ Modal/ Toast/
        <Name>.tsx  <Name>.stories.tsx  <Name>.test.tsx  <Name>.module.css
    console/
      ConsolePane.tsx           three-pane layout, used by all seven kinds
      SchemaTreePanel.tsx       left, foldable, lazy-expanding
      QueryEditor.tsx           CodeMirror 6 + CMD+Enter
      RecentQueries.tsx         own history dropdown, from /api/history
      ResultGrid.tsx            bottom pane, self-contained, internal scroll
      Pager.tsx                 Prev/Next/First, page number, opt-in row count
      ReadOnlyNotice.tsx        renders a rejected statement and why
    engines/                only per-engine knowledge in the whole frontend
      mysql.ts postgres.ts cassandra.ts redis.ts etcd.ts types.ts
    projects/               ProjectSwitcher, ConnectionForm, ConnectionList
    audit/                  AuditTable, AuditFilters, AuditDetail
    users/                  UserTable, UserForm, PasswordForm
    lib/                    api.ts, types.ts, session.ts
    pages/
      index.astro  login.astro  console/[connection].astro
      projects.astro  audit-trail.astro  users.astro
  .storybook/
```

The modularity requirement is enforced by `engines/` being the only place a database name appears in the frontend. `ConsolePane` receives an `EngineDescriptor` and never branches on kind.

### 5.2 Editor

CodeMirror 6 with `@codemirror/lang-sql` for the MySQL/Postgres/Cassandra dialects, and a custom `StreamLanguage` for the Redis and etcd command grammars. Autocomplete is fed from the live `SchemaTree` — table and column names come from the same fetch that populates the left panel, so completions are always real. `Mod-Enter` runs the query (`Mod` maps to CMD on macOS, Ctrl elsewhere). Write keywords are highlighted in the error color as a hint before the server rejects them.

`RecentQueries` sits in the editor toolbar: the caller's own last 20 distinct statements for the current connection, from `/api/history`. Picking one loads it into the editor without running it — a history dropdown that executes on click is how someone accidentally re-runs a 40-second scan.

### 5.3 Audit trail UI (`/audit-trail`)

Admin-only. A filterable table over `audit_log`, **grouped by `query_id`** so a six-page browse reads as one entry: who, when, which project/connection, the statement, allowed or denied (denials in terracotta with the reason), pages fetched, total elapsed ms, rows, and error if any. Expanding a group shows per-page timing. Filters for user, connection, allowed/denied and time range; server-side paging; CSV export (ungrouped — one row per audit record, since an export is for forensics). Clicking a row opens the full statement in a read-only CodeMirror instance so long SQL stays legible.

### 5.4 Palette — brown-ish light

```
--bg          #FAF7F2   parchment
--surface     #F3EDE4   panel
--border      #DFD3C3   hairline
--text        #3E322A   primary
--muted       #7A6A5D   secondary
--accent      #8B5E3C   walnut — primary actions, active tab
--accent-soft #C9A227   brass — focus ring, selection
--ok          #6B8E4E   moss
--error       #A94F3C   terracotta — denials, errors
```

Light theme only, as requested. Tokens live in `design-system/tokens/colors.ts` and are emitted as CSS custom properties, so Storybook and the app cannot drift.

## 6. The `demo/` folder

A note on naming: my standing instruction is never to use the word "demo", but you named `demo-start.sh` and `demo-stop.sh` explicitly, so I'm following your naming here. Say the word if you'd rather it be `sandbox/`.

`demo/` is a self-contained, fully seeded environment for exercising the console against real servers:

```
demo/
  podman-compose.yml      mysql, postgres, cassandra, redis, etcd
  seed/
    mysql.sql             tables with realistic rows
    postgres.sql          schemas, views, foreign keys
    cassandra.cql         keyspace with partition + clustering keys
    redis.sh              string, hash, list, set, zset, stream keys
    etcd.sh               nested /config/... and /service/... prefixes
  readonly-users.sql      SELECT-only DB users the console connects as
  demo-start.sh           up, wait, seed, register connections, print all configs
  demo-stop.sh            tear down containers, volumes and network
```

`demo-start.sh` brings the containers up with `podman-compose`, polls each for readiness in a 1-second loop, applies the seeds, creates the read-only users, then calls the console API to create a `demo` project with all seven connections already wired — so when it finishes you log in and every console works against real data. It ends by printing every credential, host, port and URL it used, so the environment is fully inspectable.

This doubles as the fixture for integration tests (7.2), so there is one seeded environment rather than two that drift.

## 7. Testing

### 7.1 Unit — every class

**Java, JUnit 6** (ships with Boot 4.1). One test class per production class, plus a smoke test asserting every `@Component`/`@Service` has one, so the "all classes" requirement can't silently rot.

Engines are tested against mocked drivers (`JdbcTemplate`, `CqlSession`, Lettuce `RedisCommands`, jetcd `KV`) — no containers, so `mvn test` stays fast.

The highest-value tests, per Rule 9, encode *why* the behavior matters:

- **Read-only guards** — the security boundary, so they get adversarial cases: `SELECT` with a trailing `; DROP TABLE`, `WITH x AS (DELETE ... RETURNING *) SELECT`, comment-obfuscated `SEL/**/ECT`, mixed case, leading whitespace/newlines, `EXPLAIN ANALYZE` (which executes), Redis `EVAL` with a writing script, etcd `put` disguised by flags.
- **`SecretCipher`** — round-trip, distinct IV per encryption, tampered ciphertext fails GCM auth rather than returning garbage.
- **`PostgresKeyStore`** — generates each key once, reuses it after restart, and two concurrent boots don't race into two different master keys (which would make already-stored secrets undecryptable).
- **`PasswordHasher`** — same password yields different hashes, verification is constant-time, wrong password fails.
- **`Jwt`** — expiry, tampered signature, wrong secret, role claim cannot be edited client-side.
- **`AuditService`** — a denied statement is still logged; a thrown engine error still logs elapsed time and the error; page 2 reuses the `query_id` of page 1 rather than opening a new one.
- **`CursorCache`** — a cursor expires after its idle window, an expired cursor raises rather than returning stale rows, cursors are released when the pool is evicted, and one user cannot fetch another user's cursor by guessing its id.
- **`HistoryController`** — returns only the caller's statements; a forged user parameter is ignored rather than honored.
- **Parsers** — `RedisCommandParser` and `EtcdCommandParser` quoting/escaping/malformed input; `PrefixTreeBuilder` `/`-splitting.
- **`ConnectionRegistry`** — pool reuse, eviction on config change, no pool leak on connect failure.

**TypeScript, Jest** with `@swc/jest` and jsdom, matching the reference project's setup, `@testing-library/react`, one `.test.tsx` beside each component. Covers `ConsolePane` keyboard handling (CMD+Enter fires, plain Enter does not), `SchemaTreePanel` fold/unfold and lazy-load, `Pager` (Next disabled when `hasMore` is false, Prev hidden on page 1, `410` offering re-run rather than showing stale rows), `RecentQueries` loading a statement into the editor **without** executing it, `ReadOnlyNotice` rendering a denial, audit filters building the right query string, `api.ts` error/401/403/410 paths, and every design-system component's states.

### 7.2 Integration — `it.sh` only, never in the build

Tagged `@Tag("integration-test")`. `pom.xml` sets `<excluded.groups>integration-test</excluded.groups>`, so `mvn test` and `mvn package` skip them; `it.sh` flips it with `-Dexcluded.groups= -Dincluded.groups=integration-test`. Exactly the reference project's mechanism.

`it.sh` reuses `demo/` for its environment, then runs tagged tests covering: schema listing per engine, query execution per engine, **paging across all seven engines against real data** (page 2 continues where page 1 ended, with no gaps or repeats — the failure mode that silently corrupts what an operator believes they're looking at), cursor expiry returning `410`, **write statements rejected against real servers** (the layer-2 and layer-3 check — a write must fail even with the guard bypassed in the test), auth rejection, role enforcement, audit rows written for both allowed and denied statements, secrets round-tripping through Postgres encrypted, and config surviving a backend restart.

## 8. Scripts and ports

| Script | Does |
|---|---|
| `start.sh` | starts metadata Postgres, backend and frontend, waits for health, prints `links.sh` |
| `stop.sh` | stops all three |
| `it.sh` | demo env up, run tagged integration tests, tear down |
| `links.sh` | prints every URL |
| `demo/demo-start.sh` `demo/demo-stop.sh` | seeded target servers + auto-registered connections |
| `backend/start.sh` `backend/stop.sh`, `frontend/start.sh` `frontend/stop.sh` | each alone |

All bash, no comments, no emoji, no sleep longer than 1, readiness by polling loop — per your bash rules. Containers use `podman-compose` and a `Containerfile`.

| Service | Port |
|---|---|
| frontend (Astro) | 4321 |
| backend | 8099 |
| Storybook | 6006 |
| console metadata Postgres | 5433 |
| target MySQL / Postgres / Cassandra / Redis / etcd | 3306 / 5432 / 9042 / 6379 / 2379 |

Backend is on 8099 rather than the reference project's 8095, and the metadata Postgres on 5433, so this can run alongside both the reference project and the `demo/` Postgres.

`links.sh` prints:

```
http://localhost:4321/               console
http://localhost:4321/projects       project and connection config
http://localhost:4321/audit-trail    audit trail
http://localhost:4321/users          user management
http://localhost:8099/swagger        swagger ui
http://localhost:8099/v3/api-docs    openapi json
http://localhost:8099/actuator/health
http://localhost:6006/               storybook
```

## 9. Security posture

What this design commits to:

- **Read-only everywhere**, enforced in three independent layers (4.2). No UI toggle to disable it.
- **Secrets encrypted at rest** with AES-256-GCM, so they never appear in a `SELECT`, a screenshot or a log line. Per 3.2 the key lives in the same database, which makes this **obfuscation against casual disclosure, not a boundary against anyone who obtains the database**. Listed here as a real property, described as what it is.
- **Real user accounts** with PBKDF2 password hashing, two roles, no hardcoded credential — `admin/admin` is a first-boot seed the UI nags you to change.
- **Full audit trail** of every statement, allowed or denied, append-only, admin-visible at `/audit-trail`.
- **Backend binds `127.0.0.1`** by default.
- Secrets are never returned by any API, never logged, and never included in audit rows.

What it does **not** do, stated so the boundary is explicit rather than accidental:

- **No protection against anyone who obtains the metadata database** — a `pg_dump` carries both the ciphertext and the key, so every stored connection password is recoverable (3.2). This is the single biggest gap and it is a deliberate simplicity trade; the fix is a one-class swap to an env-var key.
- No TLS between browser and backend. Loopback-only deployment is the mitigation; a shared host needs a TLS terminator in front.
- No rate limiting on login. Brute-forcing `admin/admin` from localhost is possible until the password is changed.
- No audit-log tamper-evidence (hash chaining). Append-only grants stop the application from rewriting history; they do not stop a Postgres superuser.
- No MFA, no SSO, no per-connection ACLs — any `user` can query any connection in any project.

## 10. Build order

1. Backend skeleton — Boot 4.1.0, Java 25, virtual threads, metadata Postgres + `schema.sql`, OpenAPI, health.
2. Crypto — `PostgresKeyStore`, `MasterKeyProvider`, `SecretCipher`, unit tests. Everything else depends on it.
3. Users and auth — `PasswordHasher`, `Jwt`, `AuthFilter`, `UserService`, `BootstrapAdmin`, unit tests.
4. Projects and connections — repositories with encrypted secrets, `ProjectController`, unit tests.
5. Audit — `AuditService`, `AuditRepository`, `AuditController`, wrapped around every query.
6. `Engine` SPI + `ConnectionRegistry` (dynamic Hikari pools) + `CursorCache` + `JdbcEngine` + `SqlReadOnlyGuard` (MySQL and Postgres both land here).
7. `CassandraEngine`, `RedisEngine`, `EtcdEngine` + parsers, guards and per-engine paging, unit tests each.
8. `demo/` — containers, seeds, read-only users, `demo-start.sh` / `demo-stop.sh`.
9. Frontend design system + Storybook + tokens, component tests.
10. `ConsolePane`, `Pager`, `RecentQueries` and the `engines/` descriptors; wire all seven kinds.
11. Projects, users and audit-trail UIs.
12. Scripts, `podman-compose`, integration tests against `demo/`.
13. Playwright screenshots into `printscreens/`, Excalidraw-style architecture diagram, `README.md`.

## 11. Decisions log

All open questions are resolved. Recorded here so the reasoning survives.

| # | Question | Decision |
|---|---|---|
| 1 | Results: one page or pagination? | **Pagination.** Server-side, 100 rows/page, cursor-based with Prev/Next — not numbered jumps, because Cassandra paging state is forward-only (4.3). |
| 2 | Store configs in FS or Postgres? | **Postgres, everything.** Single system of record (3.1). |
| 3 | Where does the master key live? | **Postgres `keys` table**, alongside the ciphertext. SQLite dropped, one less dependency. Makes encryption obfuscation rather than a boundary — accepted knowingly (3.2, §9). |
| 4 | Static or dynamic connection pools? | **Dynamic.** One HikariCP pool per connection, built lazily, idle-evicted, failure-isolated (3.3). |
| 5 | `demo/` naming | **Keep `demo/`, `demo-start.sh`, `demo-stop.sh`** as specified, overriding the standing no-"demo" rule. |
| 6 | Metadata Postgres lifecycle | **`start.sh` owns it** as a podman container. No external Postgres required to run this. |
| 7 | Is `EXPLAIN` allowed? | **Yes.** `EXPLAIN ANALYZE` stays rejected — it executes the statement. |
| 8 | etcd grammar | **Mirror `etcdctl`** (`get --prefix /foo`), reads only. |
| 9 | Query history | **Both.** A "recent queries" dropdown in the editor (own history, loads without executing) *and* the full `/audit-trail` for admins (4.6, 5.2, 5.3). |

## 12. Not in scope — candidate follow-ups

Deliberately excluded from this build so the first version stays finishable. Listed so they're visible choices rather than forgotten ones: additional engines (MongoDB, Elasticsearch/OpenSearch, ClickHouse, Kafka, DynamoDB, S3), schema diffing across environments, saved queries shared between users, per-connection ACLs, an env-var master key, and audit-log hash chaining.

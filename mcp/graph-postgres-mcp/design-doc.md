# GraphQL-Postgres MCP - Design Document

## Overview

An MCP (Model Context Protocol) server that provides read-only access to a PostgreSQL 17 database exclusively through a GraphQL abstraction layer. PostgreSQL is never accessed directly by MCP clients. The MCP introspects all tables and columns from PostgreSQL, generates a GraphQL schema dynamically, and exposes query tools through the MCP protocol.

## Architecture

```
┌─────────────┐     MCP Protocol     ┌─────────────────────┐
│  MCP Client  │ ◄──────────────────► │   MCP Server (Node)  │
│ (Claude Code) │                     │                      │
└─────────────┘                      │  ┌─────────────────┐ │
                                     │  │  MCP Tool Layer  │ │
                                     │  │  (query, schema, │ │
                                     │  │   refresh, list) │ │
                                     │  └────────┬────────┘ │
                                     │           │          │
                                     │  ┌────────▼────────┐ │
                                     │  │  GraphQL Engine  │ │
                                     │  │  (graphql-js)    │ │
                                     │  └────────┬────────┘ │
                                     │           │          │
                                     │  ┌────────▼────────┐ │
                                     │  │ Schema Generator │ │
                                     │  │ (introspection)  │ │
                                     │  └────────┬────────┘ │
                                     │           │          │
                                     │  ┌────────▼────────┐ │
                                     │  │   pg (postgres   │ │
                                     │  │   client lib)    │ │
                                     │  └────────┬────────┘ │
                                     └───────────┼──────────┘
                                                 │ TCP 5432
                                     ┌───────────▼──────────┐
                                     │  PostgreSQL 17       │
                                     │  (podman container)  │
                                     └──────────────────────┘
```

## Core Principles

1. **No direct SQL from MCP clients** - All data access goes through the GraphQL layer inside the MCP server. The MCP tools only accept GraphQL queries.
2. **Full schema coverage** - Every table and every column in `public` schema is exposed as a GraphQL type with query resolvers.
3. **Read-only** - Only SELECT operations. No mutations, no subscriptions. GraphQL schema only contains Query root type.
4. **Auto-refresh** - The MCP provides a `refresh_schema` tool that re-introspects PostgreSQL and rebuilds the GraphQL schema. This handles new tables and new columns without restarting the server.
5. **PostgreSQL 17 in podman** - Database runs in a podman container with persistent volume.

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Runtime | Node.js (TypeScript) | MCP SDK is TypeScript-native |
| MCP SDK | @modelcontextprotocol/sdk | Official MCP server library |
| GraphQL | graphql (graphql-js) | Reference implementation, no HTTP server needed |
| Postgres Client | pg | Standard Node.js PostgreSQL driver |
| Container | Podman + PostgreSQL 17 | User requirement |
| Build | tsc | Simple TypeScript compilation |

## Schema Generation

### Introspection Query

The schema generator queries `information_schema.columns` to discover all tables and columns in the `public` schema:

```sql
SELECT table_name, column_name, data_type, is_nullable, column_default
FROM information_schema.columns
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;
```

### Type Mapping

| PostgreSQL Type | GraphQL Type |
|----------------|-------------|
| integer, smallint, bigint | Int |
| numeric, decimal, real, double precision | Float |
| boolean | Boolean |
| json, jsonb | String (JSON serialized) |
| text, varchar, char, uuid, date, timestamp, timestamptz | String |
| ARRAY types | [String] (serialized) |
| Everything else | String (fallback) |

### Generated Schema Structure

For each table `users` with columns `id (integer)`, `name (text)`, `email (text)`:

```graphql
type Users {
  id: Int
  name: String
  email: String
}

type Query {
  users(limit: Int, offset: Int): [Users]
  users_by_pk(id: Int!): Users
  list_tables: [String]
}
```

Every table gets:
- A `<table_name>` query returning `[Type]` with optional `limit` and `offset` args
- A `<table_name>_by_pk` query if the table has a primary key (args = PK columns)

## MCP Tools

### 1. `graphql_query`

Execute a GraphQL query against the auto-generated schema.

**Input:**
```json
{
  "query": "{ users(limit: 10) { id name email } }",
  "variables": {}
}
```

**Output:** GraphQL JSON response with `data` and optional `errors`.

### 2. `list_tables`

List all tables currently exposed in the GraphQL schema.

**Input:** none

**Output:** Array of table names with their column definitions.

### 3. `get_schema`

Return the current GraphQL schema in SDL format.

**Input:** none

**Output:** The full GraphQL schema definition string.

### 4. `refresh_schema`

Re-introspect PostgreSQL and rebuild the GraphQL schema. Call this after DDL changes (new tables, new columns, dropped columns).

**Input:** none

**Output:** Confirmation with count of tables and fields discovered.

## Container Setup

### podman-compose.yaml

```yaml
services:
  postgres:
    image: docker.io/library/postgres:17
    environment:
      POSTGRES_USER: graphmcp
      POSTGRES_PASSWORD: graphmcp123
      POSTGRES_DB: graphmcpdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
volumes:
  pgdata:
```

### init.sql

Seed script with sample tables to validate the MCP works:

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    published BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@test.com'),
    ('Bob', 'bob@test.com');

INSERT INTO posts (user_id, title, body, published) VALUES
    (1, 'First Post', 'Hello World', true),
    (1, 'Second Post', 'GraphQL is great', false),
    (2, 'Bobs Post', 'Testing MCP', true);

INSERT INTO tags (name) VALUES ('tech'), ('graphql'), ('mcp');
```

## Project Structure

```
graph-postgres-mcp/
├── design-doc.md
├── package.json
├── tsconfig.json
├── podman-compose.yaml
├── init.sql
├── start.sh
├── stop.sh
├── test.sh
├── install.sh                (installs MCP into Claude Code and Codex)
├── uninstall.sh              (removes MCP from Claude Code and Codex)
└── src/
    ├── index.ts              (MCP server entry point)
    ├── schema-generator.ts   (introspects PG, builds GraphQL schema)
    ├── resolvers.ts          (GraphQL resolvers that query PG)
    ├── pg-client.ts          (PostgreSQL connection pool)
    └── types.ts              (shared TypeScript types)
```

## Data Flow

### Query Flow

```
1. MCP Client sends:  graphql_query({ query: "{ users { id name } }" })
2. MCP Server receives tool call
3. GraphQL engine parses and validates query against generated schema
4. Resolver for "users" executes: SELECT id, name FROM users
5. pg client sends SQL to PostgreSQL container
6. Results flow back: PG → pg client → resolver → GraphQL → MCP response
```

### Schema Refresh Flow

```
1. MCP Client calls: refresh_schema()
2. Schema generator queries information_schema.columns
3. Schema generator queries information_schema.table_constraints (for PKs)
4. New GraphQL schema object is built from introspection results
5. Old schema is replaced atomically
6. Response confirms: "Refreshed: 3 tables, 12 fields"
```

## Configuration

Environment variables with defaults:

| Variable | Default | Purpose |
|----------|---------|---------|
| PG_HOST | localhost | PostgreSQL host |
| PG_PORT | 5432 | PostgreSQL port |
| PG_USER | graphmcp | PostgreSQL user |
| PG_PASSWORD | graphmcp123 | PostgreSQL password |
| PG_DATABASE | graphmcpdb | PostgreSQL database |

## MCP Installer (install.sh)

### Overview

`install.sh` registers this MCP server in both **Claude Code** and **OpenAI Codex CLI** so they can use the GraphQL-Postgres tools. It builds the project, then writes the MCP config into each tool's settings file.

### Claude Code Installation

Claude Code stores MCP server configs in `~/.claude.json` under the `mcpServers` key. The installer uses the `claude mcp add` CLI command:

```bash
claude mcp add graph-postgres-mcp \
  -s user \
  -- node /absolute/path/to/graph-postgres-mcp/dist/index.js
```

This registers the MCP at user scope so it is available in all projects.

### Codex CLI Installation

Codex CLI stores MCP configs in `~/.codex/config.toml`. The installer appends a `[mcp-servers.graph-postgres-mcp]` section:

```toml
[mcp-servers.graph-postgres-mcp]
type = "stdio"
command = "node"
args = ["/absolute/path/to/graph-postgres-mcp/dist/index.js"]
env = { PG_HOST = "localhost", PG_PORT = "5432", PG_USER = "graphmcp", PG_PASSWORD = "graphmcp123", PG_DATABASE = "graphmcpdb" }
```

### Installer Flow

```
1. Resolve absolute path to project directory
2. Run npm install
3. Run npm run build (tsc)
4. Detect if 'claude' CLI is available → register MCP in Claude Code
5. Detect if '~/.codex/config.toml' exists → register MCP in Codex CLI
6. Print summary of what was installed
```

### Uninstaller (uninstall.sh)

Removes the MCP registration from both tools:
- Claude Code: `claude mcp remove graph-postgres-mcp -s user`
- Codex: removes the `[mcp-servers.graph-postgres-mcp]` block from `~/.codex/config.toml`

## Limitations

- Read-only: no INSERT, UPDATE, DELETE via GraphQL
- Only `public` schema is introspected
- No GraphQL relationships/joins between tables (each table is independent)
- No authentication/authorization layer
- No pagination cursors (only limit/offset)

## Future Extensions (Not in Scope)

- Cross-table relationships via foreign key introspection
- Write mutations
- Multiple schema support
- Connection-based pagination (Relay spec)
- Subscriptions for real-time data

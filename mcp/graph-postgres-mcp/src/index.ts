import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { executeGraphQL } from "./resolvers";
import { refreshSchema, getSchemaSDL, getTableSchemas } from "./schema-generator";
import { shutdown } from "./pg-client";

const server = new McpServer({
  name: "graph-postgres-mcp",
  version: "1.0.0",
});

server.tool(
  "graphql_query",
  "Execute a read-only GraphQL query against the PostgreSQL database. All tables and columns are available as GraphQL types. Use list_tables or get_schema to discover available data.",
  {
    query: z.string().describe("GraphQL query string, e.g. { users(limit: 10) { id name email } }"),
    variables: z
      .string()
      .optional()
      .describe("Optional GraphQL variables as JSON string, e.g. {\"id\": 1}"),
  },
  async ({ query, variables }) => {
    const parsedVars = variables ? JSON.parse(variables) : undefined;
    const result = await executeGraphQL(query, parsedVars);
    return {
      content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
    };
  }
);

server.tool(
  "list_tables",
  "List all PostgreSQL tables exposed in the GraphQL schema with their columns and types.",
  {},
  async () => {
    const tables = getTableSchemas();
    const output = tables.map((t) => ({
      table: t.name,
      primaryKeys: t.primaryKeys,
      columns: t.columns.map((c) => ({
        name: c.column_name,
        type: c.data_type,
        nullable: c.is_nullable === "YES",
      })),
    }));
    return {
      content: [{ type: "text" as const, text: JSON.stringify(output, null, 2) }],
    };
  }
);

server.tool(
  "get_schema",
  "Return the current GraphQL schema in SDL format showing all available types and queries.",
  {},
  async () => {
    const sdl = getSchemaSDL();
    return {
      content: [{ type: "text" as const, text: sdl }],
    };
  }
);

server.tool(
  "refresh_schema",
  "Re-introspect PostgreSQL and rebuild the GraphQL schema. Call this after new tables or columns are added to the database.",
  {},
  async () => {
    const result = await refreshSchema();
    return {
      content: [
        {
          type: "text" as const,
          text: `Schema refreshed: ${result.tableCount} tables, ${result.fieldCount} fields`,
        },
      ],
    };
  }
);

async function main() {
  await refreshSchema();
  const transport = new StdioServerTransport();
  await server.connect(transport);

  process.on("SIGINT", async () => {
    await shutdown();
    process.exit(0);
  });

  process.on("SIGTERM", async () => {
    await shutdown();
    process.exit(0);
  });
}

main().catch((err) => {
  console.error("Failed to start MCP server:", err);
  process.exit(1);
});

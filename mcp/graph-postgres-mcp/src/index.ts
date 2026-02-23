import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { executeGraphQL } from "./resolvers";
import { refreshSchema, getSchemaSDL, getTableSchemas } from "./schema-generator";
import { shutdown } from "./pg-client";

const GRAPHQL_ONLY_NOTICE =
  "IMPORTANT: This MCP provides access to PostgreSQL ONLY through GraphQL. " +
  "You must NEVER access PostgreSQL directly with raw SQL. " +
  "Always use the graphql_query tool to read data. " +
  "Use get_schema or list_tables to discover available tables and fields. " +
  "If you need data from the database, write a GraphQL query and pass it to graphql_query.";

const server = new Server(
  { name: "graph-postgres-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "graphql_query",
      description:
        GRAPHQL_ONLY_NOTICE +
        " Execute a read-only GraphQL query against PostgreSQL. " +
        "All database tables and columns are exposed as GraphQL types. " +
        "Use list_tables first to see what is available, then build your GraphQL query.",
      inputSchema: {
        type: "object" as const,
        properties: {
          query: {
            type: "string",
            description:
              "GraphQL query string. Example: { users(limit: 10) { id name email } }",
          },
          variables: {
            type: "string",
            description:
              'Optional GraphQL variables as JSON string. Example: {"id": 1}',
          },
        },
        required: ["query"],
      },
    },
    {
      name: "list_tables",
      description:
        GRAPHQL_ONLY_NOTICE +
        " List all PostgreSQL tables currently exposed in the GraphQL schema, " +
        "including column names, types, and primary keys. " +
        "Call this first to discover what data is available before writing GraphQL queries.",
      inputSchema: {
        type: "object" as const,
        properties: {},
      },
    },
    {
      name: "get_schema",
      description:
        GRAPHQL_ONLY_NOTICE +
        " Return the full GraphQL schema in SDL format. " +
        "Shows all available types, queries, and arguments. " +
        "Use this to understand the exact GraphQL query syntax available.",
      inputSchema: {
        type: "object" as const,
        properties: {},
      },
    },
    {
      name: "refresh_schema",
      description:
        GRAPHQL_ONLY_NOTICE +
        " Re-introspect PostgreSQL and rebuild the GraphQL schema. " +
        "Call this after new tables or columns have been added to the database. " +
        "The GraphQL schema will be updated to include all new tables and fields.",
      inputSchema: {
        type: "object" as const,
        properties: {},
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "graphql_query": {
      const queryStr = args?.query as string;
      if (!queryStr) {
        return {
          content: [{ type: "text" as const, text: "Error: query parameter is required" }],
          isError: true,
        };
      }
      const variables = args?.variables
        ? JSON.parse(args.variables as string)
        : undefined;
      const result = await executeGraphQL(queryStr, variables);
      return {
        content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }],
      };
    }

    case "list_tables": {
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

    case "get_schema": {
      const sdl = getSchemaSDL();
      return {
        content: [{ type: "text" as const, text: sdl }],
      };
    }

    case "refresh_schema": {
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

    default:
      return {
        content: [
          {
            type: "text" as const,
            text: `Unknown tool: ${name}. Use graphql_query to access data through GraphQL. Never access PostgreSQL directly.`,
          },
        ],
        isError: true,
      };
  }
});

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

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const mcp_js_1 = require("@modelcontextprotocol/sdk/server/mcp.js");
const stdio_js_1 = require("@modelcontextprotocol/sdk/server/stdio.js");
const zod_1 = require("zod");
const resolvers_1 = require("./resolvers");
const schema_generator_1 = require("./schema-generator");
const pg_client_1 = require("./pg-client");
const server = new mcp_js_1.McpServer({
    name: "graph-postgres-mcp",
    version: "1.0.0",
});
server.tool("graphql_query", "Execute a read-only GraphQL query against the PostgreSQL database. All tables and columns are available as GraphQL types. Use list_tables or get_schema to discover available data.", {
    query: zod_1.z.string().describe("GraphQL query string, e.g. { users(limit: 10) { id name email } }"),
    variables: zod_1.z
        .string()
        .optional()
        .describe("Optional GraphQL variables as JSON string, e.g. {\"id\": 1}"),
}, async ({ query, variables }) => {
    const parsedVars = variables ? JSON.parse(variables) : undefined;
    const result = await (0, resolvers_1.executeGraphQL)(query, parsedVars);
    return {
        content: [{ type: "text", text: JSON.stringify(result, null, 2) }],
    };
});
server.tool("list_tables", "List all PostgreSQL tables exposed in the GraphQL schema with their columns and types.", {}, async () => {
    const tables = (0, schema_generator_1.getTableSchemas)();
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
        content: [{ type: "text", text: JSON.stringify(output, null, 2) }],
    };
});
server.tool("get_schema", "Return the current GraphQL schema in SDL format showing all available types and queries.", {}, async () => {
    const sdl = (0, schema_generator_1.getSchemaSDL)();
    return {
        content: [{ type: "text", text: sdl }],
    };
});
server.tool("refresh_schema", "Re-introspect PostgreSQL and rebuild the GraphQL schema. Call this after new tables or columns are added to the database.", {}, async () => {
    const result = await (0, schema_generator_1.refreshSchema)();
    return {
        content: [
            {
                type: "text",
                text: `Schema refreshed: ${result.tableCount} tables, ${result.fieldCount} fields`,
            },
        ],
    };
});
async function main() {
    await (0, schema_generator_1.refreshSchema)();
    const transport = new stdio_js_1.StdioServerTransport();
    await server.connect(transport);
    process.on("SIGINT", async () => {
        await (0, pg_client_1.shutdown)();
        process.exit(0);
    });
    process.on("SIGTERM", async () => {
        await (0, pg_client_1.shutdown)();
        process.exit(0);
    });
}
main().catch((err) => {
    console.error("Failed to start MCP server:", err);
    process.exit(1);
});

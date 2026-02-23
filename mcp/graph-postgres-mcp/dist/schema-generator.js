"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.refreshSchema = refreshSchema;
exports.getSchema = getSchema;
exports.getSchemaSDL = getSchemaSDL;
exports.getTableSchemas = getTableSchemas;
const graphql_1 = require("graphql");
const pg_client_1 = require("./pg-client");
let currentSchema = null;
let tableSchemas = [];
function mapPgTypeToGraphQL(pgType) {
    const t = pgType.toLowerCase();
    if (t === "integer" || t === "smallint" || t === "bigint" || t === "serial" || t === "bigserial") {
        return graphql_1.GraphQLInt;
    }
    if (t === "numeric" || t === "decimal" || t === "real" || t === "double precision") {
        return graphql_1.GraphQLFloat;
    }
    if (t === "boolean") {
        return graphql_1.GraphQLBoolean;
    }
    return graphql_1.GraphQLString;
}
function sanitizeName(name) {
    return name.replace(/[^a-zA-Z0-9_]/g, "_");
}
function capitalizeFirst(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
}
async function introspectTables() {
    const columns = (await (0, pg_client_1.query)(`SELECT table_name, column_name, data_type, is_nullable, column_default
     FROM information_schema.columns
     WHERE table_schema = 'public'
     ORDER BY table_name, ordinal_position`));
    const pks = (await (0, pg_client_1.query)(`SELECT kcu.table_name, kcu.column_name
     FROM information_schema.table_constraints tc
     JOIN information_schema.key_column_usage kcu
       ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema = kcu.table_schema
     WHERE tc.constraint_type = 'PRIMARY KEY'
       AND tc.table_schema = 'public'`));
    const pkMap = new Map();
    for (const pk of pks) {
        const existing = pkMap.get(pk.table_name) || [];
        existing.push(pk.column_name);
        pkMap.set(pk.table_name, existing);
    }
    const tableMap = new Map();
    for (const col of columns) {
        const existing = tableMap.get(col.table_name) || [];
        existing.push(col);
        tableMap.set(col.table_name, existing);
    }
    const schemas = [];
    for (const [tableName, cols] of tableMap) {
        schemas.push({
            name: tableName,
            columns: cols,
            primaryKeys: pkMap.get(tableName) || [],
        });
    }
    return schemas;
}
function buildGraphQLSchema(tables) {
    const tableTypes = new Map();
    for (const table of tables) {
        const fields = {};
        for (const col of table.columns) {
            fields[sanitizeName(col.column_name)] = {
                type: mapPgTypeToGraphQL(col.data_type),
            };
        }
        const typeName = capitalizeFirst(sanitizeName(table.name));
        tableTypes.set(table.name, new graphql_1.GraphQLObjectType({
            name: typeName,
            fields: () => fields,
        }));
    }
    const queryFields = {};
    for (const table of tables) {
        const gqlType = tableTypes.get(table.name);
        const safeName = sanitizeName(table.name);
        queryFields[safeName] = {
            type: new graphql_1.GraphQLList(gqlType),
            args: {
                limit: { type: graphql_1.GraphQLInt },
                offset: { type: graphql_1.GraphQLInt },
            },
            resolve: async (_, args) => {
                let sql = `SELECT * FROM "${table.name}"`;
                const params = [];
                if (args.limit !== undefined) {
                    params.push(args.limit);
                    sql += ` LIMIT $${params.length}`;
                }
                if (args.offset !== undefined) {
                    params.push(args.offset);
                    sql += ` OFFSET $${params.length}`;
                }
                return (0, pg_client_1.query)(sql, params);
            },
        };
        if (table.primaryKeys.length > 0) {
            const pkArgs = {};
            for (const pkCol of table.primaryKeys) {
                const colInfo = table.columns.find((c) => c.column_name === pkCol);
                const gqlPkType = colInfo ? mapPgTypeToGraphQL(colInfo.data_type) : graphql_1.GraphQLString;
                pkArgs[sanitizeName(pkCol)] = {
                    type: new graphql_1.GraphQLNonNull(gqlPkType),
                };
            }
            queryFields[`${safeName}_by_pk`] = {
                type: gqlType,
                args: pkArgs,
                resolve: async (_, args) => {
                    const whereClauses = [];
                    const params = [];
                    for (const pkCol of table.primaryKeys) {
                        params.push(args[sanitizeName(pkCol)]);
                        whereClauses.push(`"${pkCol}" = $${params.length}`);
                    }
                    const sql = `SELECT * FROM "${table.name}" WHERE ${whereClauses.join(" AND ")} LIMIT 1`;
                    const rows = await (0, pg_client_1.query)(sql, params);
                    return rows[0] || null;
                },
            };
        }
    }
    queryFields["list_tables"] = {
        type: new graphql_1.GraphQLList(graphql_1.GraphQLString),
        resolve: () => tables.map((t) => t.name),
    };
    return new graphql_1.GraphQLSchema({
        query: new graphql_1.GraphQLObjectType({
            name: "Query",
            fields: () => queryFields,
        }),
    });
}
async function refreshSchema() {
    tableSchemas = await introspectTables();
    currentSchema = buildGraphQLSchema(tableSchemas);
    const fieldCount = tableSchemas.reduce((sum, t) => sum + t.columns.length, 0);
    return { tableCount: tableSchemas.length, fieldCount };
}
function getSchema() {
    if (!currentSchema) {
        throw new Error("Schema not initialized. Call refreshSchema() first.");
    }
    return currentSchema;
}
function getSchemaSDL() {
    return (0, graphql_1.printSchema)(getSchema());
}
function getTableSchemas() {
    return tableSchemas;
}

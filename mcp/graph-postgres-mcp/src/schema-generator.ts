import {
  GraphQLSchema,
  GraphQLObjectType,
  GraphQLString,
  GraphQLInt,
  GraphQLFloat,
  GraphQLBoolean,
  GraphQLList,
  GraphQLNonNull,
  GraphQLFieldConfigMap,
  GraphQLOutputType,
  GraphQLInputType,
  GraphQLArgumentConfig,
  printSchema,
} from "graphql";
import { query } from "./pg-client";
import { ColumnInfo, PrimaryKeyInfo, TableSchema } from "./types";

let currentSchema: GraphQLSchema | null = null;
let tableSchemas: TableSchema[] = [];

function mapPgTypeToGraphQL(pgType: string): GraphQLOutputType {
  const t = pgType.toLowerCase();
  if (t === "integer" || t === "smallint" || t === "bigint" || t === "serial" || t === "bigserial") {
    return GraphQLInt;
  }
  if (t === "numeric" || t === "decimal" || t === "real" || t === "double precision") {
    return GraphQLFloat;
  }
  if (t === "boolean") {
    return GraphQLBoolean;
  }
  return GraphQLString;
}

function sanitizeName(name: string): string {
  return name.replace(/[^a-zA-Z0-9_]/g, "_");
}

function capitalizeFirst(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

async function introspectTables(): Promise<TableSchema[]> {
  const columns = (await query(
    `SELECT table_name, column_name, data_type, is_nullable, column_default
     FROM information_schema.columns
     WHERE table_schema = 'public'
     ORDER BY table_name, ordinal_position`
  )) as ColumnInfo[];

  const pks = (await query(
    `SELECT kcu.table_name, kcu.column_name
     FROM information_schema.table_constraints tc
     JOIN information_schema.key_column_usage kcu
       ON tc.constraint_name = kcu.constraint_name
       AND tc.table_schema = kcu.table_schema
     WHERE tc.constraint_type = 'PRIMARY KEY'
       AND tc.table_schema = 'public'`
  )) as PrimaryKeyInfo[];

  const pkMap = new Map<string, string[]>();
  for (const pk of pks) {
    const existing = pkMap.get(pk.table_name) || [];
    existing.push(pk.column_name);
    pkMap.set(pk.table_name, existing);
  }

  const tableMap = new Map<string, ColumnInfo[]>();
  for (const col of columns) {
    const existing = tableMap.get(col.table_name) || [];
    existing.push(col);
    tableMap.set(col.table_name, existing);
  }

  const schemas: TableSchema[] = [];
  for (const [tableName, cols] of tableMap) {
    schemas.push({
      name: tableName,
      columns: cols,
      primaryKeys: pkMap.get(tableName) || [],
    });
  }

  return schemas;
}

function buildGraphQLSchema(tables: TableSchema[]): GraphQLSchema {
  const tableTypes = new Map<string, GraphQLObjectType>();

  for (const table of tables) {
    const fields: GraphQLFieldConfigMap<unknown, unknown> = {};
    for (const col of table.columns) {
      fields[sanitizeName(col.column_name)] = {
        type: mapPgTypeToGraphQL(col.data_type),
      };
    }

    const typeName = capitalizeFirst(sanitizeName(table.name));
    tableTypes.set(
      table.name,
      new GraphQLObjectType({
        name: typeName,
        fields: () => fields,
      })
    );
  }

  const queryFields: GraphQLFieldConfigMap<unknown, unknown> = {};

  for (const table of tables) {
    const gqlType = tableTypes.get(table.name)!;
    const safeName = sanitizeName(table.name);

    queryFields[safeName] = {
      type: new GraphQLList(gqlType),
      args: {
        limit: { type: GraphQLInt },
        offset: { type: GraphQLInt },
      },
      resolve: async (_: unknown, args: { limit?: number; offset?: number }) => {
        let sql = `SELECT * FROM "${table.name}"`;
        const params: unknown[] = [];
        if (args.limit !== undefined) {
          params.push(args.limit);
          sql += ` LIMIT $${params.length}`;
        }
        if (args.offset !== undefined) {
          params.push(args.offset);
          sql += ` OFFSET $${params.length}`;
        }
        return query(sql, params);
      },
    };

    if (table.primaryKeys.length > 0) {
      const pkArgs: Record<string, GraphQLArgumentConfig> = {};
      for (const pkCol of table.primaryKeys) {
        const colInfo = table.columns.find((c) => c.column_name === pkCol);
        const gqlPkType = colInfo ? mapPgTypeToGraphQL(colInfo.data_type) : GraphQLString;
        pkArgs[sanitizeName(pkCol)] = {
          type: new GraphQLNonNull(gqlPkType as GraphQLInputType),
        };
      }

      queryFields[`${safeName}_by_pk`] = {
        type: gqlType,
        args: pkArgs,
        resolve: async (_: unknown, args: Record<string, unknown>) => {
          const whereClauses: string[] = [];
          const params: unknown[] = [];
          for (const pkCol of table.primaryKeys) {
            params.push(args[sanitizeName(pkCol)]);
            whereClauses.push(`"${pkCol}" = $${params.length}`);
          }
          const sql = `SELECT * FROM "${table.name}" WHERE ${whereClauses.join(" AND ")} LIMIT 1`;
          const rows = await query(sql, params);
          return (rows as unknown[])[0] || null;
        },
      };
    }
  }

  queryFields["list_tables"] = {
    type: new GraphQLList(GraphQLString),
    resolve: () => tables.map((t) => t.name),
  };

  return new GraphQLSchema({
    query: new GraphQLObjectType({
      name: "Query",
      fields: () => queryFields,
    }),
  });
}

export async function refreshSchema(): Promise<{ tableCount: number; fieldCount: number }> {
  tableSchemas = await introspectTables();
  currentSchema = buildGraphQLSchema(tableSchemas);
  const fieldCount = tableSchemas.reduce((sum, t) => sum + t.columns.length, 0);
  return { tableCount: tableSchemas.length, fieldCount };
}

export function getSchema(): GraphQLSchema {
  if (!currentSchema) {
    throw new Error("Schema not initialized. Call refreshSchema() first.");
  }
  return currentSchema;
}

export function getSchemaSDL(): string {
  return printSchema(getSchema());
}

export function getTableSchemas(): TableSchema[] {
  return tableSchemas;
}

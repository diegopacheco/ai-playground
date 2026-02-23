import { GraphQLSchema } from "graphql";
import { TableSchema } from "./types";
export declare function refreshSchema(): Promise<{
    tableCount: number;
    fieldCount: number;
}>;
export declare function getSchema(): GraphQLSchema;
export declare function getSchemaSDL(): string;
export declare function getTableSchemas(): TableSchema[];

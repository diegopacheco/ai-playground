"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.executeGraphQL = executeGraphQL;
const graphql_1 = require("graphql");
const schema_generator_1 = require("./schema-generator");
async function executeGraphQL(queryStr, variables) {
    const schema = (0, schema_generator_1.getSchema)();
    const result = await (0, graphql_1.graphql)({
        schema,
        source: queryStr,
        variableValues: variables,
    });
    return {
        data: result.data,
        errors: result.errors?.map((e) => ({
            message: e.message,
            locations: e.locations,
            path: e.path,
        })),
    };
}

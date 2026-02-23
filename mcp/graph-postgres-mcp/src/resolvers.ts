import { graphql } from "graphql";
import { getSchema } from "./schema-generator";

export async function executeGraphQL(
  queryStr: string,
  variables?: Record<string, unknown>
): Promise<{ data?: unknown; errors?: unknown[] }> {
  const schema = getSchema();
  const result = await graphql({
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

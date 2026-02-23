export declare function executeGraphQL(queryStr: string, variables?: Record<string, unknown>): Promise<{
    data?: unknown;
    errors?: unknown[];
}>;

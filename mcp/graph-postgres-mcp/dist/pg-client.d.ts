import { Pool } from "pg";
export declare function getPool(): Pool;
export declare function query(text: string, params?: unknown[]): Promise<unknown[]>;
export declare function shutdown(): Promise<void>;

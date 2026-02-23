import { Pool } from "pg";

let pool: Pool | null = null;

export function getPool(): Pool {
  if (!pool) {
    pool = new Pool({
      host: process.env.PG_HOST || "localhost",
      port: parseInt(process.env.PG_PORT || "5432", 10),
      user: process.env.PG_USER || "graphmcp",
      password: process.env.PG_PASSWORD || "graphmcp123",
      database: process.env.PG_DATABASE || "graphmcpdb",
    });
  }
  return pool;
}

export async function query(text: string, params?: unknown[]): Promise<unknown[]> {
  const result = await getPool().query(text, params);
  return result.rows;
}

export async function shutdown(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}

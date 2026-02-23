"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getPool = getPool;
exports.query = query;
exports.shutdown = shutdown;
const pg_1 = require("pg");
let pool = null;
function getPool() {
    if (!pool) {
        pool = new pg_1.Pool({
            host: process.env.PG_HOST || "localhost",
            port: parseInt(process.env.PG_PORT || "5432", 10),
            user: process.env.PG_USER || "graphmcp",
            password: process.env.PG_PASSWORD || "graphmcp123",
            database: process.env.PG_DATABASE || "graphmcpdb",
        });
    }
    return pool;
}
async function query(text, params) {
    const result = await getPool().query(text, params);
    return result.rows;
}
async function shutdown() {
    if (pool) {
        await pool.end();
        pool = null;
    }
}

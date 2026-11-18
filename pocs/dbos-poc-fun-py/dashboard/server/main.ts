import { postgresDb } from "./db.ts";
import { routes } from "./routes.ts";

const url = Bun.env.DATABASE_URL;
const port = Number(Bun.env.PORT);
if (!url?.startsWith("postgresql://")) throw new Error("DATABASE_URL must be a postgresql:// url");
if (!port) throw new Error("PORT is required");

const server = Bun.serve({
  hostname: "127.0.0.1",
  port,
  routes: routes(postgresDb(url)),
  fetch: request => Response.json({ error: `no route ${new URL(request.url).pathname}` }, { status: 404 }),
});

console.log(`dbos dashboard api on ${server.url}`);

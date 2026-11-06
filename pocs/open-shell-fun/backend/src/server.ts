import { runOpenshell } from "./cli";
import { OpenShell } from "./openshell";
import { buildRoutes } from "./routes";
import { terminalSocket, type TerminalData } from "./terminal";

const port = Number(process.env.BACKEND_PORT ?? 8091);
const image = process.env.SANDBOX_IMAGE ?? "localhost/open-shell-fun-sandbox:1.0";

const server = Bun.serve<TerminalData>({
  port,
  idleTimeout: 120,
  routes: buildRoutes(new OpenShell(runOpenshell, image)),
  websocket: terminalSocket,
  fetch: () => Response.json({ error: "not found" }, { status: 404 }),
});

console.log(`backend listening on ${server.url}`);

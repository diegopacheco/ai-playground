import { CliError } from "./cli";
import type { OpenShell } from "./openshell";
import type { TerminalData } from "./terminal";
import { parseRule, requireCommand, requireName, requireRuleName, ValidationError } from "./validation";

type Req = Bun.BunRequest<"/api/sandboxes/:name">;
type RuleReq = Bun.BunRequest<"/api/sandboxes/:name/rules/:rule">;

const json = (body: unknown, status = 200) => Response.json(body, { status });

async function handle(work: () => Promise<unknown>): Promise<Response> {
  try {
    return json(await work());
  } catch (error) {
    if (error instanceof ValidationError) {
      return json({ error: error.message }, 400);
    }
    if (error instanceof CliError) {
      return json({ error: error.message }, 502);
    }
    return json({ error: String(error) }, 500);
  }
}

async function body(req: Request): Promise<Record<string, unknown>> {
  try {
    return (await req.json()) as Record<string, unknown>;
  } catch {
    throw new ValidationError("body must be JSON");
  }
}

export function buildRoutes(shell: OpenShell) {
  const name = (req: Req) => requireName(req.params.name, "name");
  return {
    "/api/health": {
      GET: () => handle(async () => ({ gateway: await shell.status() })),
    },
    "/api/sandboxes": {
      GET: () => handle(() => shell.listSandboxes()),
      POST: (req: Request) => handle(async () => {
        const created = requireName((await body(req)).name, "name");
        await shell.createSandbox(created);
        return { name: created };
      }),
    },
    "/api/sandboxes/:name": {
      DELETE: (req: Req) => handle(async () => {
        await shell.deleteSandbox(name(req));
        return { deleted: req.params.name };
      }),
    },
    "/api/sandboxes/:name/exec": {
      POST: (req: Req) => handle(async () => shell.exec(name(req), requireCommand((await body(req)).command))),
    },
    "/api/sandboxes/:name/policy": {
      GET: (req: Req) => handle(async () => ({ yaml: await shell.policy(name(req)) })),
    },
    "/api/sandboxes/:name/rules": {
      POST: (req: Req) => handle(async () => ({ output: await shell.addRule(name(req), parseRule(await body(req))) })),
    },
    "/api/sandboxes/:name/rules/:rule": {
      DELETE: (req: RuleReq) => handle(async () => ({
        output: await shell.removeRule(requireName(req.params.name, "name"), requireRuleName(req.params.rule)),
      })),
    },
    "/api/sandboxes/:name/attacks": {
      POST: (req: Req) => handle(() => shell.attack(name(req))),
    },
    "/api/sandboxes/:name/terminal": (req: Req, server: Bun.Server<TerminalData>) => {
      try {
        const sandbox = name(req);
        return server.upgrade(req, { data: { name: sandbox } }) ? undefined : json({ error: "websocket upgrade required" }, 426);
      } catch (error) {
        return json({ error: String(error) }, 400);
      }
    },
    "/api/sandboxes/:name/events": {
      GET: (req: Req) => handle(() => shell.events(name(req))),
    },
  };
}

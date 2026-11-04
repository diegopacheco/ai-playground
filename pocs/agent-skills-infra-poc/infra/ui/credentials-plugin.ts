import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import type { Plugin } from "vite";

type ClientInfo = { client_id: string; kind: string; description: string; scopes: string[] };

function readSecrets(dir: string): Map<string, string> {
  const secrets = new Map<string, string>();
  for (const file of readdirSync(dir).filter((f) => f.endsWith(".env"))) {
    const values = Object.fromEntries(
      readFileSync(join(dir, file), "utf8")
        .split("\n")
        .filter((line) => line.includes("="))
        .map((line) => [line.slice(0, line.indexOf("=")), line.slice(line.indexOf("=") + 1).trim()]),
    );
    secrets.set(values.CLIENT_ID, values.CLIENT_SECRET);
  }
  return secrets;
}

async function fetchClients(authUrl: string, adminSecret: string): Promise<ClientInfo[]> {
  const basic = Buffer.from(`infra-admin:${adminSecret}`).toString("base64");
  const token = await fetch(`${authUrl}/oauth/token`, {
    method: "POST",
    headers: { authorization: `Basic ${basic}`, "content-type": "application/x-www-form-urlencoded" },
    body: "grant_type=client_credentials&scope=clients:read",
  }).then((r) => r.json() as Promise<{ access_token: string }>);
  const clients = await fetch(`${authUrl}/v1/admin/clients`, {
    headers: { authorization: `Bearer ${token.access_token}` },
  }).then((r) => r.json() as Promise<{ items: ClientInfo[] }>);
  return clients.items;
}

export function localCredentials(secretsDir: string, authUrl: string): Plugin {
  return {
    name: "agentinfra-local-credentials",
    configureServer(server) {
      server.middlewares.use("/__local/credentials", async (_req, res) => {
        try {
          const secrets = readSecrets(join(secretsDir, "credentials"));
          const clients = await fetchClients(authUrl, secrets.get("infra-admin") ?? "");
          const items = clients.map((c) => ({ ...c, client_secret: secrets.get(c.client_id) ?? "" }));
          res.setHeader("content-type", "application/json");
          res.setHeader("cache-control", "no-store");
          res.end(JSON.stringify({ items }));
        } catch (error) {
          res.statusCode = 503;
          res.end(JSON.stringify({ error: "credentials_unavailable", error_description: String(error) }));
        }
      });
    },
  };
}

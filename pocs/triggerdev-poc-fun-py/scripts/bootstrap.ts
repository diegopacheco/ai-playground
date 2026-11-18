import { createCipheriv, createHash, randomBytes } from "node:crypto";
import { execFileSync } from "node:child_process";

const ORG_TITLE = "Agent POC";
const PROJECT_NAME = "agent-workflow";
const USER_EMAIL = "agent@poc.local";
const ALPHABET = "123456789abcdefghijkmnopqrstuvwxyz";

interface Org { slug: string; title: string }
interface Project { externalRef: string; slug: string; name: string }
interface ProjectEnv { apiKey: string }

function required(name: string): string {
  const value = process.env[name] ?? "";
  if (!value) throw new Error(`${name} is required`);
  return value;
}

const apiUrl = required("TRIGGER_API_URL");
const encryptionKey = required("ENCRYPTION_KEY");
const postgresContainer = required("TRIGGER_POSTGRES_CONTAINER");

function newToken(): string {
  const bytes = randomBytes(40);
  return `tr_pat_${[...bytes].map(byte => ALPHABET[byte % ALPHABET.length]).join("")}`;
}

function encrypt(value: string): string {
  const nonce = randomBytes(12);
  const cipher = createCipheriv("aes-256-gcm", Buffer.from(encryptionKey, "utf8"), nonce);
  const ciphertext = cipher.update(value, "utf8", "hex") + cipher.final("hex");
  return JSON.stringify({ nonce: nonce.toString("hex"), ciphertext, tag: cipher.getAuthTag().toString("hex") });
}

function seedToken(): string {
  const token = newToken();
  const hashed = createHash("sha256").update(token).digest("hex");
  const sql = `
INSERT INTO "User" (id, email, "authenticationMethod", "confirmedBasicDetails", name, "displayName", "updatedAt")
VALUES ('agentpocuser', '${USER_EMAIL}', 'MAGIC_LINK', true, 'Agent POC', 'Agent POC', now())
ON CONFLICT (email) DO NOTHING;
INSERT INTO "PersonalAccessToken" (id, name, "encryptedToken", "obfuscatedToken", "hashedToken", "userId", "updatedAt")
SELECT 'agentpocpat${randomBytes(8).toString("hex")}', 'agent-poc', '${encrypt(token)}'::jsonb, '${token.slice(0, 11)}********', '${hashed}', id, now()
FROM "User" WHERE email = '${USER_EMAIL}';`;
  execFileSync("podman", ["exec", "-i", postgresContainer, "psql", "-q", "-U", "postgres", "-d", "main", "-v", "ON_ERROR_STOP=1"], { input: sql, stdio: ["pipe", "ignore", "inherit"] });
  return token;
}

async function call<T>(token: string, path: string, body?: object): Promise<T> {
  const response = await fetch(`${apiUrl}${path}`, {
    method: body ? "POST" : "GET",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!response.ok) throw new Error(`${path} returned ${response.status}: ${await response.text()}`);
  return response.json() as Promise<T>;
}

async function works(token: string): Promise<boolean> {
  if (!token) return false;
  const response = await fetch(`${apiUrl}/api/v1/orgs`, { headers: { authorization: `Bearer ${token}` } });
  return response.ok;
}

const previous = process.env.TRIGGER_ACCESS_TOKEN ?? "";
const token = (await works(previous)) ? previous : seedToken();
const orgs = await call<Org[]>(token, "/api/v1/orgs");
const org = orgs.find(item => item.title === ORG_TITLE) ?? (await call<Org>(token, "/api/v1/orgs", { title: ORG_TITLE }));
const projects = await call<Project[]>(token, `/api/v1/orgs/${org.slug}/projects`);
const project = projects.find(item => item.name === PROJECT_NAME) ?? (await call<Project>(token, `/api/v1/orgs/${org.slug}/projects`, { name: PROJECT_NAME }));
const env = await call<ProjectEnv>(token, `/api/v1/projects/${project.externalRef}/dev`);

console.log(`TRIGGER_ACCESS_TOKEN=${token}`);
console.log(`TRIGGER_PROJECT_REF=${project.externalRef}`);
console.log(`TRIGGER_SECRET_KEY=${env.apiKey}`);
console.log(`TRIGGER_ORG_SLUG=${org.slug}`);
console.log(`TRIGGER_PROJECT_SLUG=${project.slug}`);
console.log(`TRIGGER_USER_EMAIL=${USER_EMAIL}`);

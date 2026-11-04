import { clearSession, loadSession, saveSession } from "./session";
import type { ClientInfo, EvalRun, EvalRunDetail, Evaluator, LocalCredential, LogLine, LogsSummary, Me, Session, StateEntry, StateSummary } from "./types";

export class UnauthorizedError extends Error {}

type TokenResponse = { access_token: string; expires_in: number; scope: string };
type ErrorBody = { error?: string; error_description?: string };

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ErrorBody;
    return body.error_description ?? body.error ?? response.statusText;
  } catch {
    return response.statusText;
  }
}

export async function login(clientId: string, clientSecret: string): Promise<Session> {
  const response = await fetch("/api/auth/oauth/token", {
    method: "POST",
    headers: {
      "content-type": "application/x-www-form-urlencoded",
      authorization: `Basic ${btoa(`${encodeURIComponent(clientId)}:${encodeURIComponent(clientSecret)}`)}`,
    },
    body: new URLSearchParams({ grant_type: "client_credentials" }),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  const token = (await response.json()) as TokenResponse;
  const session = {
    clientId,
    accessToken: token.access_token,
    expiresAt: Date.now() + token.expires_in * 1000,
    scopes: token.scope.split(" ").filter(Boolean),
  };
  saveSession(session);
  return session;
}

export async function fetchLocalCredentials(): Promise<LocalCredential[]> {
  const response = await fetch("/__local/credentials");
  if (!response.ok) throw new Error(await errorMessage(response));
  return ((await response.json()) as { items: LocalCredential[] }).items;
}

async function get<T>(service: string, path: string): Promise<T> {
  const session = loadSession();
  if (!session) throw new UnauthorizedError("not signed in");
  const response = await fetch(`/api/${service}${path}`, { headers: { authorization: `Bearer ${session.accessToken}` } });
  if (response.status === 401) {
    clearSession();
    throw new UnauthorizedError(await errorMessage(response));
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as T;
}

function query(params: Record<string, string | undefined>): string {
  const clean = Object.entries(params).filter((entry): entry is [string, string] => Boolean(entry[1]));
  return clean.length ? `?${new URLSearchParams(clean)}` : "";
}

export const fetchRuns = (tenant = "") =>
  get<{ items: EvalRun[] }>("evals", `/v1/evals/runs${query({ limit: "500", tenant })}`).then((r) => r.items);
export const fetchMe = () => get<Me>("auth", "/v1/me");

export function fetchStateEntries(all: boolean, tenant: string, prefix: string): Promise<StateEntry[]> {
  if (all) return get<{ items: StateEntry[] }>("state", `/v1/admin/state${query({ tenant, prefix })}`).then((r) => r.items);
  return get<{ tenant: string; items: Omit<StateEntry, "tenant">[] }>("state", `/v1/state${query({ prefix })}`).then((r) =>
    r.items.map((i) => ({ ...i, tenant: r.tenant })),
  );
}

export function fetchLogLines(all: boolean, tenant: string, level: string, sinceMinutes: string): Promise<LogLine[]> {
  const params = { tenant: all ? tenant : undefined, level, since_minutes: sinceMinutes, limit: "300" };
  if (all) return get<{ items: LogLine[] }>("logs", `/v1/admin/logs${query(params)}`).then((r) => r.items);
  return get<{ tenant: string; items: Omit<LogLine, "tenant">[] }>("logs", `/v1/logs${query(params)}`).then((r) =>
    r.items.map((i) => ({ ...i, tenant: r.tenant })),
  );
}
export const fetchRun = (id: string) => get<EvalRunDetail>("evals", `/v1/evals/runs/${encodeURIComponent(id)}`);
export const fetchEvaluators = () => get<{ items: Evaluator[] }>("evals", "/v1/evals/evaluators").then((r) => r.items);
export const fetchClients = () => get<{ items: ClientInfo[] }>("auth", "/v1/admin/clients").then((r) => r.items);
export const fetchStateSummary = () => get<{ items: StateSummary[] }>("state", "/v1/admin/state/summary").then((r) => r.items);
export const fetchLogsSummary = () => get<LogsSummary>("logs", "/v1/admin/logs/summary?since_minutes=60&limit=40");

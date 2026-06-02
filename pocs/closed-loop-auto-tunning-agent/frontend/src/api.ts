import type {
  CallResult,
  CbMetrics,
  CircuitBreakerSettings,
  Scenario,
  RunSummary,
  TuneResult,
  TuneStatus,
} from "./types";

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`);
  if (!res.ok) throw new Error(`GET ${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? "" : JSON.stringify(body),
  });
  if (!res.ok) {
    let message = `POST ${path} -> ${res.status}`;
    try {
      const data = await res.json();
      if (data && data.error) message = data.error;
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const getScenario = () => getJson<Scenario>("/sim/scenario");
export const setScenario = (s: Scenario) => postJson<Scenario>("/sim/scenario", s);

export const getCbMetrics = () => getJson<CbMetrics>("/metrics/circuitbreaker");
export const getPatternMetrics = (key: string) =>
  getJson<Record<string, number>>(`/metrics/${key}`);
export const getCbConfig = () => getJson<CircuitBreakerSettings>("/config/circuitbreaker");
export const applyCbConfig = (s: CircuitBreakerSettings) =>
  postJson<unknown>("/config/circuitbreaker", s);

export const tuneCb = (run: RunSummary | null) =>
  postJson<TuneResult>("/tune/circuitbreaker", run);
export const getTuneStatus = () => getJson<TuneStatus>("/tune/status");

export const resetCb = () => fetch("/api/cb/reset", { method: "POST" });

export async function callEndpoint(path: string): Promise<CallResult> {
  const res = await fetch(`/api${path}`, { method: "POST" });
  return res.json() as Promise<CallResult>;
}

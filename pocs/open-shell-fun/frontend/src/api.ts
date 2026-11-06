import type { ExecResult, NetworkRule, PolicyEvent, Sandbox, ScenarioResult } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  const body = (await res.json()) as T & { error?: string };
  if (!res.ok) {
    throw new Error(body.error ?? `request failed with ${res.status}`);
  }
  return body;
}

const post = <T>(path: string, body: unknown) => request<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  sandboxes: () => request<Sandbox[]>("/sandboxes"),
  createSandbox: (name: string) => post<{ name: string }>("/sandboxes", { name }),
  deleteSandbox: (name: string) => request<{ deleted: string }>(`/sandboxes/${name}`, { method: "DELETE" }),
  exec: (name: string, command: string) => post<ExecResult>(`/sandboxes/${name}/exec`, { command }),
  policy: (name: string) => request<{ yaml: string }>(`/sandboxes/${name}/policy`),
  addRule: (name: string, rule: NetworkRule) => post<{ output: string }>(`/sandboxes/${name}/rules`, rule),
  removeRule: (name: string, rule: string) => request<{ output: string }>(`/sandboxes/${name}/rules/${rule}`, { method: "DELETE" }),
  events: (name: string) => request<PolicyEvent[]>(`/sandboxes/${name}/events`),
  attacks: (name: string) => post<ScenarioResult[]>(`/sandboxes/${name}/attacks`, {}),
  terminalUrl: (name: string) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/api/sandboxes/${name}/terminal`,
};

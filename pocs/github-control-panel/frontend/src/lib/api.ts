import { getToken } from "./token";
import type {
  ActionCenter,
  Dashboard,
  Insights,
  IssueDetail,
  IssueListItem,
  RepoView,
  SyncResponse,
} from "./types";

const BASE = "/api";

async function asJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(`request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  listRepos: () => fetch(`${BASE}/repos`).then(asJson<RepoView[]>),

  addRepos: (repos: string[]) =>
    fetch(`${BASE}/repos`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repos }),
    }).then(asJson<RepoView[]>),

  removeRepo: (id: number) =>
    fetch(`${BASE}/repos/${id}`, { method: "DELETE" }).then(asJson<RepoView[]>),

  sync: () => {
    const token = getToken();
    const headers: Record<string, string> = {};
    if (token) {
      headers["X-GitHub-Token"] = token;
    }
    return fetch(`${BASE}/sync`, { method: "POST", headers }).then(asJson<SyncResponse>);
  },

  dashboard: () => fetch(`${BASE}/dashboard`).then(asJson<Dashboard>),

  issues: () => fetch(`${BASE}/issues`).then(asJson<IssueListItem[]>),

  issue: (id: number) => fetch(`${BASE}/issues/${id}`).then(asJson<IssueDetail>),

  actionCenter: () => fetch(`${BASE}/action-center`).then(asJson<ActionCenter>),

  insights: () => fetch(`${BASE}/insights`).then(asJson<Insights>),
};

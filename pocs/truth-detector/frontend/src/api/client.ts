import type {
  Analysis,
  AnalyzeResponse,
  LeaderboardEntry,
  TrackedUser,
  WeeklyScore,
} from "../types";

const BASE_URL = "http://localhost:3000/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function analyzeUser(githubUser: string): Promise<AnalyzeResponse> {
  return request("/analyze", {
    method: "POST",
    body: JSON.stringify({ github_user: githubUser }),
  });
}

export function getAnalysis(id: string): Promise<Analysis> {
  return request(`/analyze/${id}`);
}

export function getLatestAnalysis(username: string): Promise<Analysis> {
  return request(`/users/${username}/latest`);
}

export function getWeeklyScores(username: string): Promise<WeeklyScore[]> {
  return request(`/users/${username}/weekly`);
}

export function getLeaderboard(): Promise<LeaderboardEntry[]> {
  return request("/leaderboard");
}

export function getLeaderboardWeek(
  weekStart: string
): Promise<LeaderboardEntry[]> {
  return request(`/leaderboard/week/${weekStart}`);
}

export function trackUser(githubUser: string): Promise<TrackedUser> {
  return request("/leaderboard/track", {
    method: "POST",
    body: JSON.stringify({ github_user: githubUser }),
  });
}

export function untrackUser(username: string): Promise<void> {
  return request(`/leaderboard/track/${username}`, {
    method: "DELETE",
  });
}

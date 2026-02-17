import type { Tweet } from "./types";

const API_BASE = "http://localhost:3001/api";

export async function fetchTweets(): Promise<Tweet[]> {
  const res = await fetch(`${API_BASE}/tweets`);
  if (!res.ok) throw new Error("Failed to fetch tweets");
  return res.json();
}

export async function createTweet(username: string, content: string): Promise<Tweet> {
  const res = await fetch(`${API_BASE}/tweets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, content }),
  });
  if (!res.ok) throw new Error("Failed to create tweet");
  return res.json();
}

export async function deleteTweet(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/tweets/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete tweet");
}

export async function likeTweet(id: string): Promise<Tweet> {
  const res = await fetch(`${API_BASE}/tweets/${id}/like`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to like tweet");
  return res.json();
}

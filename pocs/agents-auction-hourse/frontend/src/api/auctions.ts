import type { Agent, Auction } from "../types/index.ts";

const BASE_URL = "";

export async function createAuction(agents: Agent[]): Promise<Auction> {
  const res = await fetch(`${BASE_URL}/api/auctions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agents }),
  });
  if (!res.ok) throw new Error("Failed to create auction");
  return res.json();
}

export async function getAuctions(): Promise<Auction[]> {
  const res = await fetch(`${BASE_URL}/api/auctions`);
  if (!res.ok) throw new Error("Failed to fetch auctions");
  return res.json();
}

export async function getAuction(id: string): Promise<Auction> {
  const res = await fetch(`${BASE_URL}/api/auctions/${id}`);
  if (!res.ok) throw new Error("Failed to fetch auction");
  return res.json();
}

export async function getAgents(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/api/agents`);
  if (!res.ok) throw new Error("Failed to fetch agents");
  return res.json();
}

import { promises as fs } from "fs";
import path from "path";
import type { GameRecord } from "./types";

const dataDir = path.join(process.cwd(), "data");
const dataFile = path.join(dataDir, "history.json");

async function readAll(): Promise<GameRecord[]> {
  try {
    const raw = await fs.readFile(dataFile, "utf8");
    return JSON.parse(raw) as GameRecord[];
  } catch {
    return [];
  }
}

export async function listGames(): Promise<GameRecord[]> {
  const games = await readAll();
  return games.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

export async function saveGame(game: GameRecord): Promise<void> {
  const games = await readAll();
  games.push(game);
  await fs.mkdir(dataDir, { recursive: true });
  await fs.writeFile(dataFile, JSON.stringify(games, null, 2), "utf8");
}

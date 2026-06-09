import { NextResponse } from "next/server";
import { listGames, saveGame } from "@/lib/store";
import type { GameRecord } from "@/lib/types";

export async function GET() {
  const games = await listGames();
  return NextResponse.json({ games });
}

export async function POST(request: Request) {
  let game: GameRecord;
  try {
    game = (await request.json()) as GameRecord;
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  if (!game.id || !game.secret || !Array.isArray(game.turns)) {
    return NextResponse.json({ error: "Malformed game record." }, { status: 400 });
  }

  await saveGame(game);
  return NextResponse.json({ ok: true });
}

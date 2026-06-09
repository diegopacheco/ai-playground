import { NextResponse } from "next/server";
import { nextMove } from "@/lib/ai";
import type { Turn } from "@/lib/types";

export async function POST(request: Request) {
  if (!process.env.OPENAI_API_KEY) {
    return NextResponse.json(
      { error: "OPENAI_API_KEY is not set on the server." },
      { status: 500 },
    );
  }

  let turns: Turn[];
  try {
    const body = (await request.json()) as { turns?: Turn[] };
    turns = Array.isArray(body.turns) ? body.turns : [];
  } catch {
    return NextResponse.json({ error: "Invalid request body." }, { status: 400 });
  }

  try {
    const move = await nextMove(turns);
    return NextResponse.json({ move });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 502 });
  }
}

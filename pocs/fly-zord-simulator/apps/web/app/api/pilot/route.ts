import { flyDecision } from "@fly-zord/agents";
import type { Battle, Side } from "@fly-zord/engine";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

interface PilotRequest {
  readonly battle: Battle;
  readonly side: Side;
  readonly provider: string;
  readonly model: string;
}

export async function POST(request: Request): Promise<Response> {
  const body = (await request.json()) as PilotRequest;
  if (!body?.battle || (body.side !== "left" && body.side !== "right")) return Response.json({ error: "a battle and a side are required" }, { status: 400 });
  const started = Date.now();
  const decision = flyDecision(body.battle, body.side, body.provider, body.model);
  return Response.json({ ...decision, elapsedMs: Date.now() - started });
}

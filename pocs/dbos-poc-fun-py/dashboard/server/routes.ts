import type { Db } from "./db.ts";

const json = (body: unknown, status = 200) => Response.json(body, { status });

async function guarded(work: () => Promise<Response>): Promise<Response> {
  try {
    return await work();
  } catch (error) {
    return json({ error: error instanceof Error ? error.message : String(error) }, 500);
  }
}

export function routes(db: Db) {
  return {
    "/api/health": () => json({ ok: true }),
    "/api/workflows": (request: Request) =>
      guarded(async () => {
        const params = new URL(request.url).searchParams;
        return json(await db.list({ status: params.get("status") ?? "", search: (params.get("search") ?? "").trim() }));
      }),
    "/api/workflows/:id": (request: Request & { params: { id: string } }) =>
      guarded(async () => {
        const detail = await db.detail(request.params.id);
        return detail ? json(detail) : json({ error: `no workflow ${request.params.id}` }, 404);
      }),
  };
}

import { describe, expect, test } from "bun:test";
import type { Db } from "../server/db.ts";
import { decodeStored } from "../server/decode.ts";
import { routes } from "../server/routes.ts";
import type { WorkflowFilter } from "../src/types.ts";

const asks: WorkflowFilter[] = [];
const db: Db = {
  async list(filter) {
    asks.push(filter);
    return { counts: [], workflows: [] };
  },
  async detail(id) {
    if (id === "boom") throw new Error("database down");
    return null;
  },
};
const api = routes(db);
const detail = (id: string) => api["/api/workflows/:id"](Object.assign(new Request(`http://x/api/workflows/${id}`), { params: { id } }));

describe("dashboard api", () => {
  test("only JSON serializations are decoded, other formats are reported as unreadable", () => {
    expect(decodeStored('{"a":1}', "agent_json")).toEqual({ format: "agent_json", readable: true, value: { a: 1 } });
    expect(decodeStored("gAWV", "py_pickle")).toEqual({ format: "py_pickle", readable: false, value: null });
    expect(decodeStored("{oops", "portable_json").readable).toBe(false);
    expect(decodeStored(null, null)).toEqual({ format: "none", readable: true, value: null });
  });

  test("the list passes the status filter and a trimmed search to the database", async () => {
    await api["/api/workflows"](new Request("http://x/api/workflows?status=PENDING&search=%20run-1%20"));
    expect(asks.at(-1)).toEqual({ status: "PENDING", search: "run-1" });
  });

  test("an unknown workflow is a 404 and a database failure is a 500 with the reason", async () => {
    expect((await detail("nope")).status).toBe(404);
    const failed = await detail("boom");
    expect(failed.status).toBe(500);
    expect(await failed.json()).toEqual({ error: "database down" });
  });
});

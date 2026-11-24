import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { fileURLToPath } from "node:url";
import { connect, type Connection } from "../client/connect.ts";
import { RpcError } from "../client/rawRpc.ts";

const serverEntry = fileURLToPath(new URL("../src/main.ts", import.meta.url));
const FAST = { REPORT_STEP_MS: "50", TASK_POLL_INTERVAL_MS: "20", REPORT_AGENT_MODEL: "" };

let conn: Connection;

beforeAll(async () => {
  conn = await connect(serverEntry, FAST);
});

afterAll(async () => {
  await conn.close();
});

describe("skills over MCP", () => {
  test("the server advertises the skills and tasks extensions on the 2026-07-28 protocol", () => {
    expect(conn.client.getNegotiatedProtocolVersion()).toBe("2026-07-28");
    const extensions = conn.client.getServerCapabilities()?.extensions ?? {};
    expect(Object.keys(extensions).sort()).toEqual(["io.modelcontextprotocol/skills", "io.modelcontextprotocol/tasks"]);
  });

  test("both skill folders are published as skill:// resources", async () => {
    const { skills } = await conn.skills.list();
    expect(skills.map((s) => s.uri)).toEqual(["skill://market-research/SKILL.md", "skill://task-etiquette/SKILL.md"]);
  });

  test("SKILL.md and its reference file are verified against the published digest", async () => {
    const { skills } = await conn.skills.list();
    const research = skills.find((s) => s.uri === "skill://market-research/SKILL.md");
    if (!research) throw new Error("market-research skill missing");
    const main = await conn.skills.read(research);
    expect(main.verified).toBe(true);
    expect(main.frontmatter?.["name"]).toBe("market-research");
    expect(main.body).toContain("market_report");
    const style = await conn.skills.read(research, "references/report-style.md");
    expect(style.text).toContain("Never invent data");
  });
});

describe("long running tasks", () => {
  test("the tool returns a task handle at once and polling reaches completed with the report", async () => {
    const handle = await conn.tasks.callTool("market_report", { topic: "solar panels", depth: "quick" });
    expect(handle.status).toBe("working");
    expect(handle.pollIntervalMs).toBe(20);
    const seen: string[] = [];
    const done = await conn.tasks.waitUntilDone(handle, (t) => seen.push(t.status));
    expect(seen.at(-1)).toBe("completed");
    if (done.status !== "completed") throw new Error(`expected completed, got ${done.status}`);
    expect(JSON.stringify(done.result)).toContain("Market report: solar panels");
  });

  test("cancel stops a running task and it stays cancelled", async () => {
    const handle = await conn.tasks.callTool("market_report", { topic: "fax machines", depth: "deep" });
    await conn.tasks.cancel(handle.taskId);
    expect((await conn.tasks.get(handle.taskId)).status).toBe("cancelled");
    await Bun.sleep(300);
    expect((await conn.tasks.get(handle.taskId)).status).toBe("cancelled");
  });

  test("an unknown task id is a protocol error, not a fake empty task", async () => {
    expect(conn.tasks.get("does-not-exist")).rejects.toBeInstanceOf(RpcError);
  });

  test("invalid tool input is rejected before any task is created", async () => {
    const result = await conn.tasks.callTool("market_report", { topic: "x" }).catch((e: unknown) => e);
    expect(result).toBeInstanceOf(Error);
  });
});

describe("clients without the Tasks extension", () => {
  test("a legacy 2025 client still gets the full report synchronously", async () => {
    const legacy = new Client({ name: "legacy", version: "1" });
    await legacy.connect(new StdioClientTransport({ command: "bun", args: [serverEntry], env: { PATH: Bun.env.PATH ?? "", ...FAST }, stderr: "ignore" }));
    try {
      const result = await legacy.callTool({ name: "market_report", arguments: { topic: "tea", depth: "quick" } });
      expect(JSON.stringify(result.content)).toContain("Market report: tea");
    } finally {
      await legacy.close();
    }
  });
});

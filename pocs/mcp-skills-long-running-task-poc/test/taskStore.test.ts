import { describe, expect, test } from "bun:test";
import { TaskStore } from "../src/tasks/store.ts";
import { gate, manualClock, sequentialIds, silentLogger } from "./fakes.ts";

const TTL = 60000;

function newStore() {
  const clock = manualClock();
  return { clock, store: new TaskStore({ clock, ids: sequentialIds(), logger: silentLogger, ttlMs: TTL }) };
}

describe("TaskStore", () => {
  test("a task is working until its job resolves, so a client polling early never sees a result", async () => {
    const { store } = newStore();
    const g = gate();
    const task = store.start(async () => {
      await g.promise;
      return { text: "report" };
    });
    expect(store.get(task.id)?.status).toBe("working");
    g.open();
    const done = await store.settled(task.id);
    expect(done).toMatchObject({ status: "completed", output: { text: "report" } });
  });

  test("progress reports become the status message the client shows while waiting", async () => {
    const { store } = newStore();
    const g = gate();
    const task = store.start(async ({ report }) => {
      report("step 1/2");
      await g.promise;
      return { text: "x" };
    });
    await Bun.sleep(0);
    expect(store.get(task.id)?.message).toBe("step 1/2");
    g.open();
    await store.settled(task.id);
  });

  test("a job error becomes a failed task carrying the error, not a crash of the server", async () => {
    const { store } = newStore();
    const task = store.start(async () => {
      throw new Error("source unavailable");
    });
    expect(await store.settled(task.id)).toMatchObject({ status: "failed", error: "source unavailable" });
  });

  test("cancel aborts the running job and the late result never overwrites the cancellation", async () => {
    const { store } = newStore();
    const g = gate();
    let aborted = false;
    const task = store.start(async ({ signal }) => {
      signal.addEventListener("abort", () => (aborted = true));
      await g.promise;
      return { text: "too late" };
    });
    await Bun.sleep(0);
    expect(store.cancel(task.id)?.status).toBe("cancelled");
    expect(aborted).toBe(true);
    g.open();
    expect((await store.settled(task.id))?.status).toBe("cancelled");
  });

  test("a task cancelled before its job starts never runs the job", async () => {
    const { store } = newStore();
    let ran = false;
    const task = store.start(async () => {
      ran = true;
      return { text: "x" };
    });
    store.cancel(task.id);
    await store.settled(task.id);
    expect(ran).toBe(false);
    expect(store.get(task.id)?.status).toBe("cancelled");
  });

  test("cancelling a finished task keeps its result, so a slow cancel cannot destroy work", async () => {
    const { store } = newStore();
    const task = store.start(async () => ({ text: "kept" }));
    await store.settled(task.id);
    expect(store.cancel(task.id)).toMatchObject({ status: "completed", output: { text: "kept" } });
  });

  test("tasks expire after their ttl so the in-memory store cannot grow forever", async () => {
    const { store, clock } = newStore();
    const task = store.start(async () => ({ text: "x" }));
    await store.settled(task.id);
    clock.advance(TTL + 1);
    expect(store.get(task.id)).toBeUndefined();
  });

  test("unknown ids return undefined so the protocol layer can answer not found", () => {
    const { store } = newStore();
    expect(store.get("missing")).toBeUndefined();
    expect(store.cancel("missing")).toBeUndefined();
  });
});

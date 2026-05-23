import { describe, test, expect } from "bun:test";
import { withDisposable, type AsyncDisposable } from "../src/lifecycle.ts";

class FakeResource implements AsyncDisposable {
  public closed = false;
  public closeCallCount = 0;
  async close(): Promise<void> {
    this.closeCallCount += 1;
    this.closed = true;
  }
}

describe("withDisposable", () => {
  test("calls close exactly once on success", async () => {
    const resource = new FakeResource();
    const result = await withDisposable(
      async () => resource,
      async (r) => {
        expect(r.closed).toBe(false);
        return "ok";
      },
    );
    expect(result).toBe("ok");
    expect(resource.closeCallCount).toBe(1);
    expect(resource.closed).toBe(true);
  });

  test("calls close exactly once even when the body throws", async () => {
    const resource = new FakeResource();
    await expect(
      withDisposable(
        async () => resource,
        async () => {
          throw new Error("body blew up");
        },
      ),
    ).rejects.toThrow("body blew up");
    expect(resource.closeCallCount).toBe(1);
    expect(resource.closed).toBe(true);
  });

  test("re-throws the body error even if close throws", async () => {
    const resource: AsyncDisposable = {
      async close() {
        throw new Error("close blew up");
      },
    };
    await expect(
      withDisposable(
        async () => resource,
        async () => {
          throw new Error("body blew up");
        },
      ),
    ).rejects.toThrow("body blew up");
  });

  test("close errors during success path are swallowed (not silently lost — a real impl logs)", async () => {
    const resource: AsyncDisposable = {
      async close() {
        throw new Error("close blew up");
      },
    };
    const result = await withDisposable(
      async () => resource,
      async () => 42,
    );
    expect(result).toBe(42);
  });

  test("does not call close when acquire throws — the resource was never acquired", async () => {
    let useWasCalled = false;
    await expect(
      withDisposable<AsyncDisposable, void>(
        async () => {
          throw new Error("acquire failed");
        },
        async () => {
          useWasCalled = true;
        },
      ),
    ).rejects.toThrow("acquire failed");
    expect(useWasCalled).toBe(false);
  });
});

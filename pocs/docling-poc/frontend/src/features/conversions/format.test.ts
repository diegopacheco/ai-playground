import { describe, expect, test } from "bun:test";
import { acceptsFile, formatBytes, formatDuration } from "./format";

describe("acceptsFile", () => {
  const exts = [".pdf", ".docx"];
  test("matches extensions case insensitively so REPORT.PDF uploads", () => {
    expect(acceptsFile("REPORT.PDF", exts)).toBe(true);
  });
  test("rejects files without an extension or with an unsupported one", () => {
    expect(acceptsFile("README", exts)).toBe(false);
    expect(acceptsFile("a.exe", exts)).toBe(false);
  });
});

describe("formatting", () => {
  test("switches units at the boundaries", () => {
    expect(formatDuration(999)).toBe("999 ms");
    expect(formatDuration(1500)).toBe("1.5 s");
    expect(formatBytes(1023)).toBe("1023 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });
});

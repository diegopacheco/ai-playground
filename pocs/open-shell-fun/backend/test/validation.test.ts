import { describe, expect, test } from "bun:test";
import { parseRule, requireName, ValidationError } from "../src/validation";

const valid = { ruleName: "github_api", binary: "/usr/bin/curl", host: "api.github.com", port: 443, access: "read-only", enforcement: "enforce" };

describe("parseRule", () => {
  test("accepts a read only rule exactly as OpenShell expects it", () => {
    expect(parseRule(valid)).toEqual({ ...valid, access: "read-only", enforcement: "enforce" });
  });

  test("rejects a host carrying a colon because it would inject extra fields into the endpoint spec", () => {
    expect(() => parseRule({ ...valid, host: "api.github.com:443:full" })).toThrow(ValidationError);
  });

  test("rejects relative binaries because OpenShell matches binaries by absolute path", () => {
    expect(() => parseRule({ ...valid, binary: "curl" })).toThrow("binary must be an absolute path");
  });

  test("rejects unknown access presets instead of silently widening access", () => {
    expect(() => parseRule({ ...valid, access: "admin" })).toThrow("access must be one of");
  });

  test("rejects out of range ports", () => {
    expect(() => parseRule({ ...valid, port: 70000 })).toThrow("port must be between");
  });
});

describe("requireName", () => {
  test("rejects names that look like CLI flags so they cannot alter the openshell command", () => {
    expect(() => requireName("--no-keep", "name")).toThrow(ValidationError);
  });
});

import { describe, expect, test } from "bun:test";
import type { CliResult } from "../src/cli";
import { endpointSpec, OpenShell } from "../src/openshell";
import type { NetworkRule } from "../src/types";

function fakeRunner(result: CliResult) {
  const calls: string[][] = [];
  const run = async (args: string[]) => {
    calls.push(args);
    return result;
  };
  return { calls, run };
}

const rule: NetworkRule = { ruleName: "github_api", binary: "/usr/bin/curl", host: "api.github.com", port: 443, access: "read-only", enforcement: "enforce" };

describe("OpenShell", () => {
  test("a rule becomes an L7 rest endpoint so the proxy inspects every HTTP method", () => {
    expect(endpointSpec(rule)).toBe("api.github.com:443:read-only:rest:enforce");
  });

  test("adding a rule waits for the sandbox to load the new policy revision", async () => {
    const fake = fakeRunner({ code: 0, stdout: "loaded", stderr: "" });
    await new OpenShell(fake.run, "img").addRule("box", rule);
    expect(fake.calls[0]).toEqual(["policy", "update", "box", "--rule-name", "github_api", "--binary", "/usr/bin/curl", "--add-endpoint", "api.github.com:443:read-only:rest:enforce", "--wait"]);
  });

  test("exec keeps the user command after -- so it can never become an openshell flag", async () => {
    const fake = fakeRunner({ code: 7, stdout: "", stderr: "blocked" });
    const result = await new OpenShell(fake.run, "img").exec("box", "curl https://x.dev --policy evil");
    expect(fake.calls[0].slice(-4)).toEqual(["--", "sh", "-c", "curl https://x.dev --policy evil"]);
    expect(result).toEqual({ exitCode: 7, stdout: "", stderr: "blocked" });
  });

  test("a failing CLI call surfaces its stderr instead of pretending success", async () => {
    const fake = fakeRunner({ code: 1, stdout: "", stderr: "sandbox not found" });
    await expect(new OpenShell(fake.run, "img").deleteSandbox("ghost")).rejects.toThrow("sandbox not found");
  });

  test("sandboxes are created from the curl enabled image without provider prompts", async () => {
    const fake = fakeRunner({ code: 0, stdout: "", stderr: "" });
    await new OpenShell(fake.run, "localhost/img:1").createSandbox("box");
    expect(fake.calls[0]).toEqual(["sandbox", "create", "--name", "box", "--from", "localhost/img:1", "--no-auto-providers", "--detach", "--no-tty"]);
  });

  test("policy strips the metadata header and returns only the YAML document", async () => {
    const fake = fakeRunner({ code: 0, stdout: "Version: 2\nHash: abc\n---\nversion: 1\n", stderr: "" });
    expect(await new OpenShell(fake.run, "img").policy("box")).toBe("version: 1\n");
  });
});

import { describe, expect, test } from "bun:test";
import { evaluate, SCENARIOS, verdictOf } from "../src/attacks";

const scenario = (id: string) => SCENARIOS.find((s) => s.id === id)!;

describe("verdictOf", () => {
  test("curl with HTTP 000 means the proxy refused the connection before any status existed", () => {
    expect(verdictOf({ exitCode: 7, stdout: "\nHTTP 000", stderr: "" })).toBe("BLOCKED");
  });

  test("an L7 deny returns exit 0 from curl, so the 403 status must still count as blocked", () => {
    expect(verdictOf({ exitCode: 0, stdout: '{"error":"policy_denied"}\nHTTP 403', stderr: "" })).toBe("BLOCKED");
  });

  test("a successful request is allowed", () => {
    expect(verdictOf({ exitCode: 0, stdout: "Anything added dilutes everything else.\nHTTP 200", stderr: "" })).toBe("ALLOWED");
  });
});

describe("evaluate", () => {
  test("an attack that gets through is flagged insecure so the scorecard cannot hide it", () => {
    expect(evaluate(scenario("exfil"), { exitCode: 0, stdout: "\nHTTP 200", stderr: "" })).toMatchObject({ verdict: "ALLOWED", secure: false });
  });

  test("a blocked attack is secure", () => {
    expect(evaluate(scenario("userns"), { exitCode: 1, stdout: "", stderr: "Operation not permitted" }).secure).toBe(true);
  });

  test("a legit read blocked by default deny is not a security failure, only a missing rule", () => {
    expect(evaluate(scenario("github-read"), { exitCode: 7, stdout: "\nHTTP 000", stderr: "" })).toMatchObject({ verdict: "BLOCKED", secure: true });
  });

  test("every network scenario prints the HTTP status so L7 denials are detectable", () => {
    SCENARIOS.filter((s) => s.command.startsWith("curl")).forEach((s) => expect(s.command).toContain("%{http_code}"));
  });
});

import { describe, expect, test } from "bun:test";
import { parsePolicyEvents } from "../src/logs";

const LOG = [
  "[1793992739.691] [sandbox] [OCSF ] [ocsf] NET:OPEN [INFO] host.containers.internal:17670",
  "[1793992743.545] [sandbox] [OCSF ] [ocsf] NET:OPEN [MED] DENIED /usr/bin/curl(0) -> api.github.com:443 [reason:transparent_tcp_policy_denied]",
  "[1793992750.807] [sandbox] [OCSF ] [ocsf] HTTP:GET [INFO] ALLOWED GET https://api.github.com/zen [policy:github_api engine:l7]",
  "[1793992751.113] [sandbox] [OCSF ] [ocsf] HTTP:POST [MED] DENIED POST https://api.github.com/repos/octocat/hello-world/issues [policy:github_api engine:l7] [reason:L7_REQUEST deny POST api.github.com:443/repos/octocat/hello-world/issues reason=POST /repos/octocat/hello-world/issues not permitted by policy]",
  "[1793992751.116] [sandbox] [OCSF ] [ocsf] EVENT [INFO] ssh relay closed (channel_id=5049)",
].join("\n");

describe("parsePolicyEvents", () => {
  const events = parsePolicyEvents(LOG);

  test("keeps only allow and deny decisions so the audit view is not flooded with plumbing events", () => {
    expect(events.map((e) => e.decision)).toEqual(["DENIED", "ALLOWED", "DENIED"]);
  });

  test("a default deny has no matching policy, which proves nothing authorized the connection", () => {
    expect(events[0]).toMatchObject({ kind: "NET:OPEN", target: "/usr/bin/curl(0) -> api.github.com:443", policy: "-", reason: "transparent_tcp_policy_denied" });
  });

  test("an L7 deny names the rule that allowed the host but refused the method", () => {
    expect(events[2]).toMatchObject({ kind: "HTTP:POST", policy: "github_api", engine: "l7", severity: "MED" });
    expect(events[2].reason).toContain("not permitted by policy");
  });

  test("timestamps convert from seconds to epoch millis for the UI", () => {
    expect(events[1].timeMs).toBe(1793992750807);
  });
});

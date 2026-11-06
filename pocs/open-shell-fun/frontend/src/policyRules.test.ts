import { describe, expect, test } from "bun:test";
import { ruleNames } from "./policyRules";

const YAML = `version: 1
filesystem_policy:
  read_only:
    - /usr
network_policies:
  github_api:
    name: github_api
    endpoints:
      - host: api.github.com
  pypi:
    name: pypi
process:
  run_as_user: sandbox
`;

describe("ruleNames", () => {
  test("lists only network rule keys so the UI never offers to remove filesystem or process settings", () => {
    expect(ruleNames(YAML)).toEqual(["github_api", "pypi"]);
  });

  test("a default deny sandbox has no removable rules", () => {
    expect(ruleNames("version: 1\nlandlock:\n  compatibility: best_effort\n")).toEqual([]);
  });
});

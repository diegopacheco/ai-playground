export const releases = [
  { id: 1, service: "checkout", env: "prod", status: "success", at: "2026-05-30T09:14:00Z", durationMs: 184000 },
  { id: 2, service: "checkout", env: "staging", status: "success", at: "2026-05-31T11:02:00Z", durationMs: 142000 },
  { id: 3, service: "search", env: "prod", status: "failed", at: "2026-06-01T15:48:00Z", durationMs: 73000 },
  { id: 4, service: "search", env: "prod", status: "success", at: "2026-06-02T08:21:00Z", durationMs: 156000 },
  { id: 5, service: "billing", env: "prod", status: "success", at: "2026-06-03T13:37:00Z", durationMs: 201000 },
  { id: 6, service: "billing", env: "staging", status: "rolled-back", at: "2026-06-04T10:05:00Z", durationMs: 98000 },
  { id: 7, service: "search", env: "staging", status: "success", at: "2026-06-05T17:19:00Z", durationMs: 121000 },
  { id: 8, service: "checkout", env: "prod", status: "failed", at: "2026-06-06T06:44:00Z", durationMs: 64000 },
];

export const notes = `# This week
Shipped **8 releases** across three services.

- checkout moved to the new payment flow
- search hit one prod failure, fixed within the hour
- billing rolled back once in staging

> All prod incidents resolved under SLA.`;

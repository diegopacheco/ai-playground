---
name: task-etiquette
description: How to drive long running MCP tasks. Use when a tool returns a task handle, when polling task status, or when deciding to cancel a task.
---

# Task etiquette

- A task handle has a `taskId`, a `status` and a `pollIntervalMs`.
- Poll with `tasks/get` no faster than `pollIntervalMs`.
- `working` means keep going, `completed` carries the result, `failed` carries the error, `cancelled` is final.
- Tell the user the `statusMessage` while waiting, it says which step is running.
- Cancel with `tasks/cancel` when the result is no longer needed.

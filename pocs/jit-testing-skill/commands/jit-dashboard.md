---
name: jit-dashboard
description: Start the local JIT testing dashboard and open it in the user's browser. Light theme, loopback only.
---

You are running the JIT Dashboard skill. The user invoked `/jit-dashboard`.

## When to run

The cwd should contain a `.jit-testing/` directory with at least one run. If it does not, tell the user to run `/jit` first.

## What to do

1. If the user passed `--stop`, stop the dashboard:
   ```
   ~/.claude/jit/bin/jit dashboard --stop
   ```
2. Otherwise, start the dashboard:
   ```
   ~/.claude/jit/bin/jit dashboard --repo .
   ```
   The command starts a local HTTP server on `127.0.0.1`, picks a free port, opens the user's default browser, and prints the URL. Tell the user the URL.
3. Mention that the dashboard is light theme only and runs locally.

## Hard rules

- Loopback only. Never bind to `0.0.0.0` or a public interface.
- Light theme only. If the user asks for dark mode, tell them it is explicitly not supported.

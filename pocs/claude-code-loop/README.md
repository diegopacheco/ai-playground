# Claude Code /loop

The `/loop` command in Claude Code runs a prompt or slash command on a recurring interval. It defaults to every 10 minutes but can be customized.

## Syntax

```
/loop <interval> <command>
```

- `<interval>` — how often to run (e.g., `5m`, `10m`, `30s`). Defaults to `10m` if omitted.
- `<command>` — the prompt or slash command to execute repeatedly.

## Use Cases

- Poll a deployment status repeatedly
- Run tests on an interval while developing
- Monitor logs or build output periodically
- Babysit pull requests

## Running It

```bash
claude
```
Then inside the Claude Code session:

```
/loop 5m /cc-tests
```
Runs `/cc-tests` every 5 minutes.

```
/loop 2m check if the build passed on CI
```
Asks Claude to check CI build status every 2 minutes.

```
/loop 10m /commit
```
Auto-commits every 10 minutes.

```
/loop 30s are there any new errors in the logs?
```
Checks for new errors every 30 seconds.

```
/loop 5m /simplify
```
Reviews changed code for quality every 5 minutes.

## Result

The command keeps running at the specified interval until you stop the session or cancel it. Each iteration runs the full prompt/command as if you typed it fresh.

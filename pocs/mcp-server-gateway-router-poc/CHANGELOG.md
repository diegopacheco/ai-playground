# Changelog

Records the user-visible changes of the MCP server gateway router POC.

## 1.0.0 - 2026-11-05

### Added

- math-mcp upstream MCP server on port 8101 with the `add` and `multiply` tools.
- text-mcp upstream MCP server on port 8102 with the `uppercase`, `reverse` and `word_count` tools.
- Gateway MCP server on port 8100 that lists the tools of every upstream as `<upstream>__<tool>` and routes each tool call to the upstream that owns it.
- Upstreams are configured with the `UPSTREAMS` env var as `name=url` pairs.
- An upstream that is down drops out of the gateway tool list, and calling one of its tools returns an error naming the unreachable URL.
- `GET /health` on all three servers.
- Scripts `setup.sh`, `start-all.sh`, `stop-all.sh`, `status.sh` and `test-all.sh`, with ports in `scripts/ports.env`.

### Verified

- `./scripts/test-all.sh` ran 16 tests: 16 passed, 0 failed, 0 skipped.
- With `./scripts/start-all.sh` running, `tools/list` on the gateway returned `math__add math__multiply text__uppercase text__reverse text__word_count`.
- `math__multiply` with 6 and 7 returned `42`, and `text__reverse` with `gateway` returned `yawetag`.
- `start-all.sh`, `stop-all.sh` and `setup.sh` ran twice in a row with exit 0, `status.sh` exited 0 from a nested folder, and all three ports went UP and then DOWN.
